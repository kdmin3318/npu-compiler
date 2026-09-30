// C++ 커스텀 op 구현 (OrtCustomOp C++ 래퍼 API 사용).
//
// 등록/스키마 관련 부분(이 파일의 InvertedResidualOp, RegisterCustomOps)은
// 전부 boilerplate입니다 - ONNX Runtime이 우리 op을 찾고 실행하게
// 연결해주는 "의식(ceremony)"일 뿐, 컴파일러 로직이 아닙니다.
//
// 직접 채워야 하는 부분은 InvertedResidualKernel::Compute() 하나입니다.
// npu_compiler/verify_pyop.py의 _conv2d/_relu6/inverted_residual_op가
// 이미 정확성 검증까지 끝난 참조 구현이니, 그 로직을 C++로 그대로
// 옮기시면 됩니다.

#include "onnxruntime_cxx_api.h"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace {

// naive 참조 구현 (verify_pyop.py의 _conv2d와 동일한 로직).
// 속도 최적화 안 함 - 이번 단계 목표는 "정확하게 동작하는 것".
void Conv2D(
    const float* input, int64_t cin, int64_t h, int64_t w,
    const float* weight, const float* bias,
    int64_t cout, int64_t kh, int64_t kw,
    int64_t stride, int64_t groups,
    std::vector<float>& output, int64_t& out_h, int64_t& out_w) {
    int64_t pad = kh / 2;
    out_h = (h + 2 * pad - kh) / stride + 1;
    out_w = (w + 2 * pad - kw) / stride + 1;
    output.assign(static_cast<size_t>(cout * out_h * out_w), 0.0f);

    int64_t out_per_group = cout / groups;
    int64_t in_per_group = cin / groups;

    for (int64_t g = 0; g < groups; ++g) {
        for (int64_t oc = 0; oc < out_per_group; ++oc) {
            int64_t co = g * out_per_group + oc;
            for (int64_t oh = 0; oh < out_h; ++oh) {
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    float sum = bias[co];
                    for (int64_t ic = 0; ic < in_per_group; ++ic) {
                        int64_t ci = g * in_per_group + ic;
                        for (int64_t i = 0; i < kh; ++i) {
                            for (int64_t j = 0; j < kw; ++j) {
                                int64_t ih = oh * stride + i - pad;
                                int64_t iw = ow * stride + j - pad;
                                if (ih < 0 || ih >= h || iw < 0 || iw >= w) continue;
                                float in_val = input[ci * h * w + ih * w + iw];
                                float w_val = weight[co * in_per_group * kh * kw + ic * kh * kw + i * kh + j];
                                sum += in_val * w_val;
                            }
                        }
                    }
                    output[co * out_h * out_w + oh * out_w + ow] = sum;
                }
            }
        }
    }
}

void Relu6(std::vector<float>& x) {
    for (float& v : x) {
        v = std::min(std::max(v, 0.0f), 6.0f);
    }
}

struct InvertedResidualKernel {
    InvertedResidualKernel(const OrtApi& api, const OrtKernelInfo* info) {
        Ort::ConstKernelInfo kernel_info{info};
        block_type_ = kernel_info.GetAttribute<std::string>("block_type");
        dw_stride_ = kernel_info.GetAttribute<int64_t>("dw_stride");
    }

    void Compute(OrtKernelContext* context) {
        Ort::KernelContext ctx(context);

        // 입력 순서는 prepare_cpp_model.py가 만드는 그대로입니다
        // (Type A도 항상 7자리를 채우되, 없는 expand 자리는 빈 입력("")):
        //   [0] block_input
        //   [1] expand_weight (Type A면 빈 입력)
        //   [2] expand_bias   (Type A면 빈 입력)
        //   [3] dw_weight
        //   [4] dw_bias
        //   [5] proj_weight
        //   [6] proj_bias
        //   즉 Type A든 B/C든 dw_weight/dw_bias/proj_weight/proj_bias의
        //   위치는 항상 [3],[4],[5],[6]으로 동일합니다 - block_type_로
        //   분기해야 하는 건 "expand([1],[2])를 쓸지 말지"뿐입니다.

        auto block_input = ctx.GetInput(0); //텐서 받기(타입을 auto로 자동)
        const float* input_data = block_input.GetTensorData<float>(); //실제 float* 포인터 받기

        auto dw_weight = ctx.GetInput(3);
        const float* dw_weight_data = dw_weight.GetTensorData<float>();
        auto dw_bias = ctx.GetInput(4);
        const float* dw_bias_data = dw_bias.GetTensorData<float>();
        auto proj_weight = ctx.GetInput(5);
        const float* proj_weight_data = proj_weight.GetTensorData<float>();
        auto proj_bias = ctx.GetInput(6);
        const float* proj_bias_data = proj_bias.GetTensorData<float>();

        auto input_shape = block_input.GetTensorTypeAndShapeInfo().GetShape();  // [1, C, H, W]
        auto dw_weight_shape = dw_weight.GetTensorTypeAndShapeInfo().GetShape();  // [Cdw, 1, KH, KW]
        auto proj_weight_shape = proj_weight.GetTensorTypeAndShapeInfo().GetShape();  // [Cout, Cin, 1, 1]

        // 현재까지 흐른 데이터(expand 유무에 따라 갱신됨)
        const float* cur_data = input_data;
        int64_t cur_c = input_shape[1];
        int64_t cur_h = input_shape[2];
        int64_t cur_w = input_shape[3];
        std::vector<float> expand_out;  // expand 안 쓰면 비어있음 (cur_data가 input_data 그대로)

        if (block_type_ != "A") {
            auto expand_weight = ctx.GetInput(1);
            const float* expand_weight_data = expand_weight.GetTensorData<float>();
            auto expand_bias = ctx.GetInput(2);
            const float* expand_bias_data = expand_bias.GetTensorData<float>();
            auto expand_weight_shape = expand_weight.GetTensorTypeAndShapeInfo().GetShape();

            int64_t exp_h, exp_w;
            Conv2D(cur_data, cur_c, cur_h, cur_w, expand_weight_data, expand_bias_data,
                   expand_weight_shape[0], 1, 1, /*stride=*/1, /*groups=*/1, expand_out, exp_h, exp_w);
            Relu6(expand_out);

            cur_data = expand_out.data();
            cur_c = expand_weight_shape[0];
            cur_h = exp_h;
            cur_w = exp_w;
        }

        // depthwise: group 수 = 채널 수 (weight shape의 [0]번째)
        std::vector<float> dw_out;
        int64_t dw_h, dw_w;
        Conv2D(cur_data, cur_c, cur_h, cur_w, dw_weight_data, dw_bias_data,
               dw_weight_shape[0], dw_weight_shape[2], dw_weight_shape[3],
               dw_stride_, /*groups=*/dw_weight_shape[0], dw_out, dw_h, dw_w);
        Relu6(dw_out);

        // project: 1x1, group=1, 활성화 없음 (linear bottleneck)
        std::vector<float> proj_out;
        int64_t proj_h, proj_w;
        Conv2D(dw_out.data(), dw_weight_shape[0], dw_h, dw_w, proj_weight_data, proj_bias_data,
               proj_weight_shape[0], 1, 1, /*stride=*/1, /*groups=*/1, proj_out, proj_h, proj_w);

        if (block_type_ == "C") {
            // residual: block_input을 그대로 더함 (shape이 같아야 함)
            for (size_t i = 0; i < proj_out.size(); ++i) {
                proj_out[i] += input_data[i];
            }
        }

        std::vector<int64_t> output_shape = {1, proj_weight_shape[0], proj_h, proj_w};
        Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
        float* output_data = output.GetTensorMutableData<float>();
        std::memcpy(output_data, proj_out.data(), proj_out.size() * sizeof(float));
    }

private:
    std::string block_type_;
    int64_t dw_stride_ = 1;
};

struct InvertedResidualOp : Ort::CustomOpBase<InvertedResidualOp, InvertedResidualKernel> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new InvertedResidualKernel(api, info);
    }

    const char* GetName() const { return "Inverted_residual"; }
    const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }

    size_t GetInputTypeCount() const { return 7; }
    ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    // index 1, 2 (expand weight/bias)는 Type A 블록엔 없을 수 있어서
    // optional로 표시합니다. ONNX 규칙상 "중간" optional 입력은 생략이
    // 아니라 빈 문자열("")로 자리만 채워서 넘겨야 합니다.
    OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t index) const {
        if (index == 1 || index == 2) {
            return INPUT_OUTPUT_OPTIONAL;
        }
        return INPUT_OUTPUT_REQUIRED;
    }

    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
};

}  // namespace

static InvertedResidualOp c_InvertedResidualOp;

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
    const OrtApi* ort_api = api_base->GetApi(ORT_API_VERSION);
    static const char* domain = "com.npu_compiler";

    OrtStatus* status = nullptr;
    OrtCustomOpDomain* custom_op_domain = nullptr;
    status = ort_api->CreateCustomOpDomain(domain, &custom_op_domain);
    if (status != nullptr) return status;

    status = ort_api->CustomOpDomain_Add(custom_op_domain, &c_InvertedResidualOp);
    if (status != nullptr) return status;

    return ort_api->AddCustomOpDomain(options, custom_op_domain);
}
