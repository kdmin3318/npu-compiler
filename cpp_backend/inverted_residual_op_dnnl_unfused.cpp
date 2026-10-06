// "unfused + oneDNN" 베이스라인 (2차: 버퍼 재사용 + 가중치 최적 레이아웃 적용).
//
// 목적: fusion 효과를 깨끗하게 분리해서 측정하기 위한 대조군.
// inverted_residual_op.cpp(naive 3중 for문)과 비교해 "conv 연산 자체는
// oneDNN(MLAS급 SIMD/스레딩)을 쓰되, 각 단계(expand/depthwise/project)
// 결과를 여전히 전체 크기 버퍼에 완전히 materialize"하는 방식입니다.
// 즉 fusion(타일링)은 아직 적용 안 한 상태 - 나중에 만들 "fused + oneDNN"
// 버전과 비교해야 fusion 자체의 순수 효과가 드러납니다.
//
// 1차 버전(전역 shape 캐시) 대비 이번에 추가한 것:
//   1) 출력 버퍼를 매 호출마다 새로 할당하지 않고, 커널 인스턴스(=블록 하나)에
//      귀속된 버퍼를 재사용 (ORT의 메모리 플래너가 원본 그래프에서 하는 일을
//      흉내냄).
//   2) 가중치 레이아웃을 oihw/goihw로 고정하지 않고 oneDNN이 선호하는
//      내부 블록 레이아웃(format_tag::any)을 쓰도록 하고, 그 변환(reorder)은
//      가중치가 변하지 않으므로 커널 인스턴스당 딱 1번만 수행 후 캐시.
//
// 주의: shape 캐시를 전역(static map)으로 두면 안 됨 - MobileNetV2에는
// shape은 같지만 가중치 값은 다른 블록이 여러 개 있어서(같은 스테이지 내
// 반복 블록), 가중치 reorder 결과를 shape 기준 전역 캐시에 두면 다른
// 블록의 가중치를 잘못 재사용하는 버그가 됨. 그래서 이번 버전은 캐시를
// 커널 인스턴스(= ONNX 그래프의 블록 노드 하나, CreateKernel마다 생성됨)
// 멤버로 둠 - 전역 캐시보다 코드가 번거로워지는 대신 정확성이 보장됨.
//
// op 스키마는 inverted_residual_op.cpp와 완전히 동일.

#include "onnxruntime_cxx_api.h"
#include "oneapi/dnnl/dnnl.hpp"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace {

using dnnl::memory;

dnnl::engine& GetEngine() {
    static dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    return eng;
}
dnnl::stream& GetStream() {
    static dnnl::stream strm(GetEngine());
    return strm;
}

// conv 하나(expand/depthwise/project 중 하나)에 해당하는 상태.
// - primitive/메모리 디스크립터: 첫 Setup()에서 1회 생성, 이후 재사용
// - 가중치: 최적 레이아웃으로 1회 reorder 후 캐시 (가중치 값은 추론 내내 불변)
// - 출력 버퍼: 크기가 변하지 않으므로 1회 할당 후 재사용
struct DnnlConvStage {
    bool initialized = false;
    dnnl::convolution_forward prim;
    memory::desc src_md;           // 블록 경계 레이아웃 (plain NCHW, ORT 텐서와 직접 맞닿음)
    memory::desc weights_md_plain;  // ONNX 원본 가중치 레이아웃 (OIHW/GOIHW)
    memory::desc weights_md_opt;    // oneDNN이 선호하는 레이아웃 (any로부터 결정됨)
    memory::desc bias_md;
    memory::desc dst_md;
    bool weights_need_reorder = false;
    bool weight_cached = false;
    std::vector<float> reordered_weight_buf;
    std::vector<float> output_buf;
    int64_t out_h = 0, out_w = 0;

    void Setup(int64_t cin, int64_t h, int64_t w, int64_t cout, int64_t kh, int64_t kw,
               int64_t stride, int64_t groups) {
        auto& eng = GetEngine();
        int64_t pad = kh / 2;
        out_h = (h + 2 * pad - kh) / stride + 1;
        out_w = (w + 2 * pad - kw) / stride + 1;
        output_buf.assign(static_cast<size_t>(cout * out_h * out_w), 0.0f);

        memory::dims src_dims = {1, cin, h, w};
        memory::dims dst_dims = {1, cout, out_h, out_w};
        memory::dims bias_dims = {cout};
        memory::dims strides_dims = {stride, stride};
        memory::dims padding_dims = {pad, pad};

        src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
        dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);
        bias_md = memory::desc(bias_dims, memory::data_type::f32, memory::format_tag::x);

        memory::dims weights_dims;
        if (groups == 1) {
            weights_dims = {cout, cin, kh, kw};
            weights_md_plain = memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::oihw);
        } else {
            int64_t out_per_group = cout / groups;
            int64_t in_per_group = cin / groups;
            weights_dims = {groups, out_per_group, in_per_group, kh, kw};
            weights_md_plain = memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::goihw);
        }
        memory::desc weights_md_any(weights_dims, memory::data_type::f32, memory::format_tag::any);

        auto conv_pd = dnnl::convolution_forward::primitive_desc(
            eng, dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_auto,
            src_md, weights_md_any, bias_md, dst_md,
            strides_dims, padding_dims, padding_dims);

        prim = dnnl::convolution_forward(conv_pd);
        weights_md_opt = conv_pd.weights_desc();
        weights_need_reorder = (weights_md_opt != weights_md_plain);
        initialized = true;
    }

    const float* Run(const float* input, const float* weight, const float* bias) {
        auto& eng = GetEngine();
        auto& strm = GetStream();

        if (!weight_cached) {
            if (weights_need_reorder) {
                memory plain_mem(weights_md_plain, eng, const_cast<float*>(weight));
                reordered_weight_buf.assign(weights_md_opt.get_size() / sizeof(float), 0.0f);
                memory opt_mem(weights_md_opt, eng, reordered_weight_buf.data());
                dnnl::reorder(plain_mem, opt_mem).execute(strm, plain_mem, opt_mem);
                strm.wait();
            } else {
                size_t n = weights_md_plain.get_size() / sizeof(float);
                reordered_weight_buf.assign(weight, weight + n);
            }
            weight_cached = true;
        }

        memory src_mem(src_md, eng, const_cast<float*>(input));
        memory weights_mem(weights_md_opt, eng, reordered_weight_buf.data());
        memory bias_mem(bias_md, eng, const_cast<float*>(bias));
        memory dst_mem(dst_md, eng, output_buf.data());

        prim.execute(strm, {
            {DNNL_ARG_SRC, src_mem},
            {DNNL_ARG_WEIGHTS, weights_mem},
            {DNNL_ARG_BIAS, bias_mem},
            {DNNL_ARG_DST, dst_mem},
        });
        strm.wait();
        return output_buf.data();
    }
};

void Relu6(float* data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        data[i] = std::min(std::max(data[i], 0.0f), 6.0f);
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

        auto block_input = ctx.GetInput(0);
        const float* input_data = block_input.GetTensorData<float>();

        auto dw_weight = ctx.GetInput(3);
        const float* dw_weight_data = dw_weight.GetTensorData<float>();
        auto dw_bias = ctx.GetInput(4);
        const float* dw_bias_data = dw_bias.GetTensorData<float>();
        auto proj_weight = ctx.GetInput(5);
        const float* proj_weight_data = proj_weight.GetTensorData<float>();
        auto proj_bias = ctx.GetInput(6);
        const float* proj_bias_data = proj_bias.GetTensorData<float>();

        auto input_shape = block_input.GetTensorTypeAndShapeInfo().GetShape();
        auto dw_weight_shape = dw_weight.GetTensorTypeAndShapeInfo().GetShape();
        auto proj_weight_shape = proj_weight.GetTensorTypeAndShapeInfo().GetShape();

        const float* cur_data = input_data;
        int64_t cur_c = input_shape[1];
        int64_t cur_h = input_shape[2];
        int64_t cur_w = input_shape[3];

        if (block_type_ != "A") {
            auto expand_weight = ctx.GetInput(1);
            const float* expand_weight_data = expand_weight.GetTensorData<float>();
            auto expand_bias = ctx.GetInput(2);
            const float* expand_bias_data = expand_bias.GetTensorData<float>();
            auto expand_weight_shape = expand_weight.GetTensorTypeAndShapeInfo().GetShape();

            if (!expand_.initialized) {
                expand_.Setup(cur_c, cur_h, cur_w, expand_weight_shape[0], 1, 1, /*stride=*/1, /*groups=*/1);
            }
            const float* expand_out = expand_.Run(cur_data, expand_weight_data, expand_bias_data);
            Relu6(const_cast<float*>(expand_out), expand_.output_buf.size());

            cur_data = expand_out;
            cur_c = expand_weight_shape[0];
            cur_h = expand_.out_h;
            cur_w = expand_.out_w;
        }

        if (!dw_.initialized) {
            dw_.Setup(cur_c, cur_h, cur_w, dw_weight_shape[0], dw_weight_shape[2], dw_weight_shape[3],
                      dw_stride_, /*groups=*/dw_weight_shape[0]);
        }
        const float* dw_out = dw_.Run(cur_data, dw_weight_data, dw_bias_data);
        Relu6(const_cast<float*>(dw_out), dw_.output_buf.size());

        if (!proj_.initialized) {
            proj_.Setup(dw_weight_shape[0], dw_.out_h, dw_.out_w, proj_weight_shape[0], 1, 1,
                        /*stride=*/1, /*groups=*/1);
        }
        const float* proj_out = proj_.Run(dw_out, proj_weight_data, proj_bias_data);

        size_t proj_size = proj_.output_buf.size();
        std::vector<int64_t> output_shape = {1, proj_weight_shape[0], proj_.out_h, proj_.out_w};
        Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
        float* output_data = output.GetTensorMutableData<float>();

        if (block_type_ == "C") {
            for (size_t i = 0; i < proj_size; ++i) {
                output_data[i] = proj_out[i] + input_data[i];
            }
        } else {
            std::memcpy(output_data, proj_out, proj_size * sizeof(float));
        }
    }

private:
    std::string block_type_;
    int64_t dw_stride_ = 1;
    DnnlConvStage expand_, dw_, proj_;
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
