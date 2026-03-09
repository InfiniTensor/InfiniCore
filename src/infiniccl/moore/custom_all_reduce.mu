// Adapted from: https://github.com/vllm-project/vllm/blob/v0.8.2/csrc/custom_all_reduce.cu
// #include "torch_musa/csrc/aten/musa/Exceptions.h"
// #include "torch_musa/csrc/core/MUSAGuard.h"
// #include "torch_musa/csrc/core/MUSAStream.h"

#include <mccl.h>
#include <musa_bf16.h>
#include "custom_all_reduce.muh"

// Fake pointer type, must match fptr_t type in ops.h.
// We use this type alias to indicate when pointers are passed in as int64_t.
using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

extern "C" fptr_t init_custom_ar(const std::vector<fptr_t>& fake_ipc_ptrs, void* rank_data, size_t rank_data_sz, int64_t rank, bool full_nvlink) {
  int world_size = fake_ipc_ptrs.size();
  if (world_size > 8) throw std::invalid_argument("world size > 8 is not supported");
  if (world_size % 2 != 0) throw std::invalid_argument("Odd num gpus is not supported for now");
  if (rank < 0 || rank >= world_size) throw std::invalid_argument("invalid rank passed in");

  sglang::Signal* ipc_ptrs[8];

  for (int i = 0; i < world_size; i++) {
    ipc_ptrs[i] = reinterpret_cast<sglang::Signal*>(fake_ipc_ptrs[i]);
  }

  return (fptr_t) new sglang::CustomAllreduce(
      ipc_ptrs, rank_data, rank_data_sz, rank, world_size, full_nvlink);
}

/**
 * Make sure tensor t's data lies completely within ((char)t.data_ptr()) +
 * t.numel() * t.element_size(). This is slightly weaker than t.is_contiguous()
 * because it allows transpose of contiguous slice (i.e. slicing the first
 * dimension). Currently, we require this because stride information is not
 * passed into the kernels and we treat input tensors as flat.
 *
 * Examples
 * A = torch.zeros(3, 3, 3)
 * 1. A: OK
 * 2. A[1:]: OK
 * 3. A.permute(2, 0, 1): OK
 * 4. A[1:].permute(2, 0, 1): OK
 * 5. A[None].expand(2, -1, -1, -1): Not OK
 * 6. A[:, 1:, 1:]: Not OK
 */
// bool _is_weak_contiguous(torch::Tensor& t) {
//   return t.is_contiguous() ||
//          (t.storage().nbytes() - t.storage_offset() * t.element_size() == t.numel() * t.element_size());
// }

/**
 * Performs an out-of-place allreduce and stores result in out.
 *
 * If _reg_buffer is null, assumes inp.data_ptr() is already IPC-registered.
 * Otherwise, _reg_buffer is assumed to be IPC-registered and inp is first
 * copied into _reg_buffer.
 */
extern "C" void all_reduce(fptr_t _fa, void* inp, void* out, size_t rank_data_sz, mcclDataType_t datatype, fptr_t _reg_buffer, int64_t reg_buffer_sz_bytes, musaStream_t stream) {
  auto fa = reinterpret_cast<sglang::CustomAllreduce*>(_fa);
  // const at::musa::OptionalMUSAGuard device_guard(device_of(inp));
  // auto stream = c10::musa::getCurrentMUSAStream().stream();

  // TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  // TORCH_CHECK_EQ(inp.numel(), out.numel());
  // TORCH_CHECK(_is_weak_contiguous(out));
  // TORCH_CHECK(_is_weak_contiguous(inp));
  // auto input_size = inp.numel() * inp.element_size();
  size_t rank_data_element_sz = 0;
  switch (datatype) {
    case mcclFloat:
      rank_data_element_sz = 4;
      break;
    case mcclHalf:
      rank_data_element_sz = 2;
      break;
    case mcclBfloat16:
      rank_data_element_sz = 2;
      break;
    default:
      throw std::runtime_error("custom allreduce only supports float32, float16 and bfloat16");
  }

  auto input_size = rank_data_sz * rank_data_element_sz;
  auto reg_buffer = reinterpret_cast<void*>(_reg_buffer);

  if (reg_buffer) {
    // TORCH_CHECK_LE(input_size, reg_buffer_sz_bytes);
    CHECK_MUSA_SUCCESS(musaMemcpyAsync(reg_buffer, inp, input_size, musaMemcpyDeviceToDevice, stream));
  } else {
    reg_buffer = inp;
  }

  switch (datatype) {
    case mcclFloat: {
      fa->allreduce<float>(
          stream, reinterpret_cast<float*>(reg_buffer), reinterpret_cast<float*>(out), (int)rank_data_sz);
      break;
    }
    case mcclHalf: {
      fa->allreduce<half>(
          stream, reinterpret_cast<half*>(reg_buffer), reinterpret_cast<half*>(out), (int)rank_data_sz);
      break;
    }
    case mcclBfloat16: {
      fa->allreduce<__mt_bfloat16>(
          stream, reinterpret_cast<__mt_bfloat16*>(reg_buffer), reinterpret_cast<__mt_bfloat16*>(out), (int)rank_data_sz);
      break;
    }
    default:
      throw std::runtime_error("custom allreduce only supports float32, float16 and bfloat16");
  }
}

void dispose(fptr_t _fa) {
  delete reinterpret_cast<sglang::CustomAllreduce*>(_fa);
}

extern "C" int64_t meta_size() {
  return sizeof(sglang::Signal);
}

extern "C" void register_buffer(fptr_t _fa, const std::vector<fptr_t>& fake_ipc_ptrs) {
  auto fa = reinterpret_cast<sglang::CustomAllreduce*>(_fa);
  // TORCH_CHECK(fake_ipc_ptrs.size() == fa->world_size_);
  void* ipc_ptrs[8];
  for (int i = 0; i < fake_ipc_ptrs.size(); i++) {
    ipc_ptrs[i] = reinterpret_cast<void*>(fake_ipc_ptrs[i]);
  }
  fa->register_buffer(ipc_ptrs);
}

// Use vector<int64_t> to represent byte data for python binding compatibility.
std::tuple<std::vector<int64_t>, std::vector<int64_t>> get_graph_buffer_ipc_meta(fptr_t _fa) {
  auto fa = reinterpret_cast<sglang::CustomAllreduce*>(_fa);
  auto [handle, offsets] = fa->get_graph_buffer_ipc_meta();
  std::vector<int64_t> bytes(handle.begin(), handle.end());
  return std::make_tuple(bytes, offsets);
}

// Use vector<int64_t> to represent byte data for python binding compatibility.
void register_graph_buffers(
    fptr_t _fa, const std::vector<std::vector<int64_t>>& handles, const std::vector<std::vector<int64_t>>& offsets) {
  auto fa = reinterpret_cast<sglang::CustomAllreduce*>(_fa);
  std::vector<std::string> bytes;
  bytes.reserve(handles.size());
  for (int i = 0; i < handles.size(); i++) {
    bytes.emplace_back(handles[i].begin(), handles[i].end());
  }
  bytes.reserve(handles.size());
  fa->register_graph_buffers(bytes, offsets);
}
