// "fused, naive 타일링" - oneDNN 없이, halo 타일링 구조는 그대로 두고
// 타일 계산 자체는 naive Conv2D(inverted_residual_op.cpp와 동일 로직)로
// 수행한다.
//
// 배경: inverted_residual_op_dnnl_fused.cpp(oneDNN + 타일링)이 unfused+
// oneDNN보다 오히려 느렸음 - 타일이 작아지면서 oneDNN 호출 횟수가 폭증,
// 호출 1번당 고정 오버헤드(스레드풀 동기화 등)가 누적된 게 원인으로
// 진단됨(타일 크기를 키워 타일링을 사실상 비활성화하면 unfused+oneDNN과
// 거의 같아짐을 실측으로 확인). MCUNetV2/TinyEngine 등 실제 타일 추론
// 구현체들도 범용 라이브러리를 타일마다 부르지 않고 가벼운 전용 코드를
// 쓰는 것으로 확인됨 - 이 파일은 그 방향(naive, 호출 오버헤드 없음)을
// 시도해본다.
//
// op 스키마는 다른 변형들과 동일 (domain=com.npu_compiler, name=Inverted_residual).

#include "onnxruntime_cxx_api.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// naive 참조 구현(inverted_residual_op.cpp와 동일) + pad를 명시적으로 받도록
// 변경 - 타일링에서는 패딩을 ExtractPaddedTile이 직접 처리하므로 conv
// 자체는 pad=0으로 불러야 함 (안 그러면 타일 경계에서 이웃 타일의 실제
// 데이터 대신 0을 쓰게 되는 버그가 생김 - oneDNN 버전에서 겪었던 것과 동일한 함정).
void Conv2D(
    const float* input, int64_t cin, int64_t h, int64_t w,
    const float* weight, const float* bias,
    int64_t cout, int64_t kh, int64_t kw,
    int64_t stride, int64_t groups, int64_t pad,
    std::vector<float>& output, int64_t& out_h, int64_t& out_w) {
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

void Relu6(float* data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        data[i] = std::min(std::max(data[i], 0.0f), 6.0f);
    }
}

void ExtractPaddedTile(const float* src, int64_t C, int64_t H, int64_t W,
                        int64_t y_start, int64_t x_start, int64_t tile_h, int64_t tile_w,
                        std::vector<float>& dst) {
    dst.assign(static_cast<size_t>(C * tile_h * tile_w), 0.0f);
    for (int64_t c = 0; c < C; ++c) {
        for (int64_t ty = 0; ty < tile_h; ++ty) {
            int64_t sy = y_start + ty;
            if (sy < 0 || sy >= H) continue;
            const float* src_row = src + (c * H + sy) * W;
            float* dst_row = dst.data() + (c * tile_h + ty) * tile_w;
            for (int64_t tx = 0; tx < tile_w; ++tx) {
                int64_t sx = x_start + tx;
                if (sx < 0 || sx >= W) continue;
                dst_row[tx] = src_row[sx];
            }
        }
    }
}

void ZeroOutOfBounds(float* data, int64_t C, int64_t tile_h, int64_t tile_w,
                      int64_t valid_y0, int64_t valid_y1, int64_t valid_x0, int64_t valid_x1) {
    for (int64_t c = 0; c < C; ++c) {
        for (int64_t ty = 0; ty < tile_h; ++ty) {
            float* row = data + (c * tile_h + ty) * tile_w;
            if (ty < valid_y0 || ty >= valid_y1) {
                std::memset(row, 0, static_cast<size_t>(tile_w) * sizeof(float));
                continue;
            }
            for (int64_t tx = 0; tx < valid_x0; ++tx) row[tx] = 0.0f;
            for (int64_t tx = valid_x1; tx < tile_w; ++tx) row[tx] = 0.0f;
        }
    }
}

struct InvertedResidualKernel {
    InvertedResidualKernel(const OrtApi& api, const OrtKernelInfo* info) {
        Ort::ConstKernelInfo kernel_info{info};
        block_type_ = kernel_info.GetAttribute<std::string>("block_type");
        dw_stride_ = kernel_info.GetAttribute<int64_t>("dw_stride");
    }

    void LazySetup(int64_t cin, int64_t h, int64_t c_expand) {
        if (setup_done_) return;

        int64_t oh = (h - 1) / dw_stride_ + 1;
        int64_t c_for_budget = (block_type_ == "A") ? cin : c_expand;

        // dnnl_fused와 동일: 패치(halo 포함)의 expand 출력 버퍼가 예산 안에 들어가도록
        // P를 먼저 구하고, 그 P로 depthwise를 통과했을 때의 출력 T를 역산한다.
        const char* budget_env = std::getenv("INVRES_BUDGET_BYTES");
        int64_t budget = budget_env ? std::strtoll(budget_env, nullptr, 10) : 2621440;  // 기본 2.5MB (L2)
        int64_t p_budget = static_cast<int64_t>(
            std::sqrt(static_cast<double>(budget) / (4.0 * static_cast<double>(c_for_budget))));
        T_ = (p_budget >= 3) ? std::min<int64_t>((p_budget - 3) / dw_stride_ + 1, oh) : 1;
        T_ = std::max<int64_t>(1, T_);

        halo_in_ = (T_ - 1) * dw_stride_ + 3;
        oh_ = oh;
        ow_ = oh;
        setup_done_ = true;
    }

    void Compute(OrtKernelContext* context) {
        Ort::KernelContext ctx(context);

        auto block_input = ctx.GetInput(0);
        const float* input_data = block_input.GetTensorData<float>();
        auto input_shape = block_input.GetTensorTypeAndShapeInfo().GetShape();
        int64_t cin = input_shape[1], h = input_shape[2], w = input_shape[3];

        const float* expand_weight_data = nullptr;
        const float* expand_bias_data = nullptr;
        int64_t c_expand = cin;
        if (block_type_ != "A") {
            auto expand_weight = ctx.GetInput(1);
            expand_weight_data = expand_weight.GetTensorData<float>();
            auto expand_bias = ctx.GetInput(2);
            expand_bias_data = expand_bias.GetTensorData<float>();
            c_expand = expand_weight.GetTensorTypeAndShapeInfo().GetShape()[0];
        }

        auto dw_weight = ctx.GetInput(3);
        const float* dw_weight_data = dw_weight.GetTensorData<float>();
        auto dw_bias = ctx.GetInput(4);
        const float* dw_bias_data = dw_bias.GetTensorData<float>();
        auto proj_weight = ctx.GetInput(5);
        const float* proj_weight_data = proj_weight.GetTensorData<float>();
        auto proj_bias = ctx.GetInput(6);
        const float* proj_bias_data = proj_bias.GetTensorData<float>();
        int64_t cout = proj_weight.GetTensorTypeAndShapeInfo().GetShape()[0];

        LazySetup(cin, h, c_expand);

        std::vector<int64_t> output_shape = {1, cout, oh_, ow_};
        Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
        float* output_data = output.GetTensorMutableData<float>();

        int64_t dw_in_c = (block_type_ == "A") ? cin : c_expand;
        std::vector<float> input_tile, expand_out, dw_out, proj_out;
        int64_t tmp_h, tmp_w;

        for (int64_t oy = 0; oy < oh_; oy += T_) {
            int64_t actual_th = std::min<int64_t>(T_, oh_ - oy);
            int64_t in_y_start = oy * dw_stride_ - 1;

            for (int64_t ox = 0; ox < ow_; ox += T_) {
                int64_t actual_tw = std::min<int64_t>(T_, ow_ - ox);
                int64_t in_x_start = ox * dw_stride_ - 1;

                ExtractPaddedTile(input_data, cin, h, w, in_y_start, in_x_start, halo_in_, halo_in_, input_tile);

                const float* dw_input = input_tile.data();
                if (block_type_ != "A") {
                    Conv2D(input_tile.data(), cin, halo_in_, halo_in_, expand_weight_data, expand_bias_data,
                           c_expand, 1, 1, /*stride=*/1, /*groups=*/1, /*pad=*/0, expand_out, tmp_h, tmp_w);
                    Relu6(expand_out.data(), expand_out.size());

                    int64_t valid_y0 = std::max<int64_t>(0, -in_y_start);
                    int64_t valid_y1 = std::min<int64_t>(halo_in_, h - in_y_start);
                    int64_t valid_x0 = std::max<int64_t>(0, -in_x_start);
                    int64_t valid_x1 = std::min<int64_t>(halo_in_, w - in_x_start);
                    if (valid_y0 > 0 || valid_y1 < halo_in_ || valid_x0 > 0 || valid_x1 < halo_in_) {
                        ZeroOutOfBounds(expand_out.data(), c_expand, halo_in_, halo_in_,
                                        valid_y0, valid_y1, valid_x0, valid_x1);
                    }
                    dw_input = expand_out.data();
                }

                Conv2D(dw_input, dw_in_c, halo_in_, halo_in_, dw_weight_data, dw_bias_data,
                       dw_in_c, 3, 3, dw_stride_, /*groups=*/dw_in_c, /*pad=*/0, dw_out, tmp_h, tmp_w);
                Relu6(dw_out.data(), dw_out.size());

                Conv2D(dw_out.data(), dw_in_c, T_, T_, proj_weight_data, proj_bias_data,
                       cout, 1, 1, /*stride=*/1, /*groups=*/1, /*pad=*/0, proj_out, tmp_h, tmp_w);

                for (int64_t c = 0; c < cout; ++c) {
                    for (int64_t ty = 0; ty < actual_th; ++ty) {
                        const float* proj_row = proj_out.data() + (c * T_ + ty) * T_;
                        float* out_row = output_data + (c * oh_ + (oy + ty)) * ow_ + ox;
                        if (block_type_ == "C") {
                            const float* residual_row = input_data + (c * h + (oy + ty)) * w + ox;
                            for (int64_t tx = 0; tx < actual_tw; ++tx) {
                                out_row[tx] = proj_row[tx] + residual_row[tx];
                            }
                        } else {
                            std::memcpy(out_row, proj_row, static_cast<size_t>(actual_tw) * sizeof(float));
                        }
                    }
                }
            }
        }
    }

private:
    std::string block_type_;
    int64_t dw_stride_ = 1;
    bool setup_done_ = false;
    int64_t T_ = 0, oh_ = 0, ow_ = 0, halo_in_ = 0;
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
