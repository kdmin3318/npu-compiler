// "fused + oneDNN" - 진짜 타일(halo) 단위 스트리밍 fusion 구현.
//
// unfused+oneDNN(inverted_residual_op_dnnl_unfused.cpp) 버전은 conv는
// oneDNN을 쓰지만 여전히 expand/depthwise/project 각 단계 결과를 "블록
// 전체 크기" 버퍼에 완전히 materialize했음 - 그래서 DRAM 트래픽 관점에서는
// 사실상 unfused랑 동일한 메모리 접근 패턴이었음.
//
// 이 버전은 출력을 작은 타일(T x T) 단위로 나눠서, 타일 하나를
// expand->depthwise->project까지 끝까지 계산한 뒤 바로 최종 출력에 쓴다.
// 타일 사이 중간 결과(expand_tile/dw_tile/proj_tile)는 전부 작은
// 고정 크기 버퍼라 캐시 안에 머무를 수 있음 - 전체 크기로 materialize되는
// 건 block_input(한 번 읽기)과 최종 output(한 번 쓰기)뿐.
//
// 핵심 아이디어 (halo/겹침 타일링):
//   - 1x1 conv(expand/project)는 receptive field를 넓히지 않으므로
//     halo가 필요한 지점은 depthwise(3x3) 한 곳뿐.
//   - 출력 타일 T x T를 만들려면, depthwise 직전 입력(expand 출력 또는
//     block_input)에서 (T-1)*stride+3 크기의 "halo 포함 타일"이 필요함.
//   - 이 halo 포함 타일은 oneDNN conv 프리미티브에 pad=0으로 넣는다 -
//     패딩을 프리미티브가 하게 두면 타일 경계에서 "가짜 0"을 쓰게 되므로,
//     패딩은 우리가 ExtractPaddedTile에서 "진짜 이미지 경계에서만" 직접 처리.
//
// 타일 크기 T는 블록마다 동적으로 계산 (목표 작업셋이 캐시에 들어가게):
//     P = sqrt(budget / (4 * C_for_budget))   (halo 포함 패치 한 변)
//     T = (P - 3) / stride + 1, 그리고 OH를 넘지 않게 clamp.
//     budget은 환경변수 INVRES_BUDGET_BYTES로 지정 (기본 2.5MB = 타겟 CPU의 L2).
//   -> 후반 블록(해상도 작음)은 clamp에 걸려 자동으로 "타일 없음"(T=OH)으로
//      수렴하므로 별도 분기 없이 하나의 공식으로 양쪽 다 처리됨.
//
// op 스키마는 다른 변형들과 동일 (domain=com.npu_compiler, name=Inverted_residual).

#include "onnxruntime_cxx_api.h"
#include "oneapi/dnnl/dnnl.hpp"

#include <algorithm>
#include <cstdlib>
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

// 타일(고정 크기) 전용 conv 스테이지. unfused 버전의 DnnlConvStage와 거의
// 같은 구조(primitive 1회 생성 + 가중치 1회 reorder 캐시)지만, pad를
// 명시적으로 받는다 - 타일 경계의 "진짜 패딩"은 우리가 직접 처리하므로
// depthwise 타일 프리미티브는 pad=0으로 만들어야 함 (안 그러면 타일
// 경계에서 이웃 타일의 실제 데이터 대신 0을 쓰게 되는 버그가 생김).
struct DnnlConvStage {
    bool initialized = false;
    dnnl::convolution_forward prim;
    memory::desc src_md, weights_md_plain, weights_md_opt, bias_md, dst_md;
    bool weights_need_reorder = false;
    bool weight_cached = false;
    std::vector<float> reordered_weight_buf;
    std::vector<float> output_buf;
    int64_t out_h = 0, out_w = 0;

    void Setup(int64_t cin, int64_t h, int64_t w, int64_t cout, int64_t kh, int64_t kw,
               int64_t stride, int64_t groups, int64_t pad) {
        auto& eng = GetEngine();
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

// src(C x H x W)에서 [y_start, y_start+tile_h) x [x_start, x_start+tile_w)
// 영역을 dst(C x tile_h x tile_w)로 복사. 이미지 진짜 경계를 벗어나는
// 위치는 0으로 채움 (halo 타일링의 "패딩은 우리가 직접" 원칙).
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

// 원본(non-tiled) 모델에서는 depthwise가 "expand의 출력"을 0으로 패딩한다.
// 근데 우리는 block_input(=expand 입력)을 먼저 0-패딩해서 expand에 넣는데,
// 1x1 conv는 선형이라 expand(0) = bias (0이 아님!) 가 나옴. 그래서 expand를
// 돌린 뒤, 진짜 이미지 경계 밖에 해당하는 위치를 명시적으로 다시 0으로
// 덮어써야 depthwise 입장에서 "진짜 0 패딩"을 보게 된다. (block_type=="A"는
// expand가 없어서 이 문제 자체가 없음 - block_input을 depthwise가 직접 보므로
// ExtractPaddedTile의 0-패딩이 곧바로 맞는 값.)
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

    // 블록 shape을 보고 타일 크기/스테이지들을 1회 셋업 (첫 Compute() 호출 때).
    void LazySetup(int64_t cin, int64_t h, int64_t w, int64_t c_expand, int64_t cout) {
        if (setup_done_) return;

        int64_t oh = (h - 1) / dw_stride_ + 1;  // depthwise pad=1,k=3 "same" 기준 전체 출력 크기
        int64_t c_for_budget = (block_type_ == "A") ? cin : c_expand;

        // 패치(expand 입력, halo 포함)의 expand 출력 버퍼가 예산 안에 들어가도록 P를 먼저 구하고,
        // 그 P로 depthwise를 통과했을 때의 출력 T를 역산한다.
        const char* budget_env = std::getenv("INVRES_BUDGET_BYTES");
        int64_t budget = budget_env ? std::strtoll(budget_env, nullptr, 10) : 2621440;  // 기본 2.5MB (L2)
        int64_t p_budget = static_cast<int64_t>(
            std::sqrt(static_cast<double>(budget) / (4.0 * static_cast<double>(c_for_budget))));
        T_ = (p_budget >= 3) ? std::min<int64_t>((p_budget - 3) / dw_stride_ + 1, oh) : 1;
        T_ = std::max<int64_t>(1, T_);

        int64_t halo_in = (T_ - 1) * dw_stride_ + 3;  // depthwise 직전 입력에 필요한 최소 패치 크기 (예산 이하)

        if (block_type_ != "A") {
            // expand: 1x1, stride1, pad0 - halo 포함 타일 크기 그대로 유지한 채 채널만 확장
            expand_.Setup(cin, halo_in, halo_in, c_expand, 1, 1, /*stride=*/1, /*groups=*/1, /*pad=*/0);
        }
        int64_t dw_in_c = (block_type_ == "A") ? cin : c_expand;
        // depthwise: halo 포함 타일(halo_in x halo_in) -> 정확히 T x T로 줄어듦, pad=0 (패딩은 이미 처리됨)
        dw_.Setup(dw_in_c, halo_in, halo_in, dw_in_c, 3, 3, dw_stride_, /*groups=*/dw_in_c, /*pad=*/0);
        // project: 1x1, stride1, pad0, T x T -> T x T
        proj_.Setup(dw_in_c, T_, T_, cout, 1, 1, /*stride=*/1, /*groups=*/1, /*pad=*/0);

        oh_ = oh;
        ow_ = oh;  // MobileNetV2 feature map은 항상 정사각형(H==W) 가정 - 기존 코드와 동일 전제
        halo_in_ = halo_in;
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

        LazySetup(cin, h, w, c_expand, cout);

        std::vector<int64_t> output_shape = {1, cout, oh_, ow_};
        Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
        float* output_data = output.GetTensorMutableData<float>();

        std::vector<float> input_tile;  // halo 포함, block_input 채널 수 기준

        for (int64_t oy = 0; oy < oh_; oy += T_) {
            int64_t actual_th = std::min<int64_t>(T_, oh_ - oy);
            int64_t in_y_start = oy * dw_stride_ - 1;  // depthwise pad=1 기준 halo 시작 위치

            for (int64_t ox = 0; ox < ow_; ox += T_) {
                int64_t actual_tw = std::min<int64_t>(T_, ow_ - ox);
                int64_t in_x_start = ox * dw_stride_ - 1;

                ExtractPaddedTile(input_data, cin, h, w, in_y_start, in_x_start, halo_in_, halo_in_, input_tile);

                const float* dw_input = input_tile.data();
                if (block_type_ != "A") {
                    const float* expand_out = expand_.Run(input_tile.data(), expand_weight_data, expand_bias_data);
                    Relu6(const_cast<float*>(expand_out), expand_.output_buf.size());

                    // depthwise가 봐야 하는 "진짜 0 패딩"을 expand 이후에 다시 적용
                    // (expand(0) = bias != 0 이라 expand 전에 패딩한 걸로는 부족함).
                    int64_t valid_y0 = std::max<int64_t>(0, -in_y_start);
                    int64_t valid_y1 = std::min<int64_t>(halo_in_, h - in_y_start);
                    int64_t valid_x0 = std::max<int64_t>(0, -in_x_start);
                    int64_t valid_x1 = std::min<int64_t>(halo_in_, w - in_x_start);
                    if (valid_y0 > 0 || valid_y1 < halo_in_ || valid_x0 > 0 || valid_x1 < halo_in_) {
                        ZeroOutOfBounds(const_cast<float*>(expand_out), c_expand, halo_in_, halo_in_,
                                        valid_y0, valid_y1, valid_x0, valid_x1);
                    }
                    dw_input = expand_out;
                }

                const float* dw_out = dw_.Run(dw_input, dw_weight_data, dw_bias_data);
                Relu6(const_cast<float*>(dw_out), dw_.output_buf.size());

                const float* proj_out = proj_.Run(dw_out, proj_weight_data, proj_bias_data);

                // proj_out은 항상 T_ x T_ (pad=0로 정확히 맞춰 설계함) - 경계 타일은
                // actual_th/actual_tw 만큼만 유효해서 그 부분만 최종 출력에 씀.
                for (int64_t c = 0; c < cout; ++c) {
                    for (int64_t ty = 0; ty < actual_th; ++ty) {
                        const float* proj_row = proj_out + (c * T_ + ty) * T_;
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
