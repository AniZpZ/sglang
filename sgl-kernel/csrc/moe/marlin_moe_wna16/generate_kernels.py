import glob
import itertools
import os
import subprocess

import jinja2

FILE_HEAD = """
// auto generated by generate.py
// clang-format off
#include "kernel.h"
#include "marlin_template.h"
namespace MARLIN_NAMESPACE_NAME {
""".strip()

TEMPLATE = ("template __global__ void Marlin<"
            "{{scalar_t}}, "
            "{{w_type_id}}, "
            "{{threads}}, "
            "{{thread_m_blocks}}, "
            "{{thread_n_blocks}}, "
            "{{thread_k_blocks}}, "
            "{{'true' if m_block_size_8 else 'false'}}, "
            "{{stages}}, "
            "{{'true' if has_act_order else 'false'}}, "
            "{{'true' if has_zp else 'false'}}, "
            "{{group_blocks}}, "
            "{{'true' if is_zp_float else 'false'}}>"
            "( MARLIN_KERNEL_PARAMS );")

# int8 with zero point case (sglang::kU8) is also supported,
# we don't add it to reduce wheel size.
SCALAR_TYPES = ["sglang::kU4", "sglang::kU4B8", "sglang::kU8B128"]
THREAD_CONFIGS = [(128, 128, 256), (64, 256, 256), (64, 128, 128)]

THREAD_M_BLOCKS = [0.5, 1, 2, 3, 4]
# group_blocks:
#   = 0 : act order case
#   = -1 : channelwise quantization
#   > 0 : group_size=16*group_blocks
GROUP_BLOCKS = [0, -1, 2, 4, 8]
DTYPES = ["fp16", "bf16"]


def remove_old_kernels():
    for filename in glob.glob(os.path.dirname(__file__) + "/kernel_*.cu"):
        subprocess.call(["rm", "-f", filename])


def generate_new_kernels():
    for scalar_type, dtype in itertools.product(SCALAR_TYPES, DTYPES):
        has_zp = "B" not in scalar_type
        all_template_str_list = []

        for group_blocks, m_blocks, thread_configs in itertools.product(
                GROUP_BLOCKS, THREAD_M_BLOCKS, THREAD_CONFIGS):

            has_act_order = group_blocks == 0
            if has_zp and has_act_order:
                continue
            if thread_configs[2] == 256:
                if m_blocks <= 1 and thread_configs[0] != 128:
                    continue
                if m_blocks > 1 and thread_configs[0] != 64:
                    continue

            k_blocks = thread_configs[0] // 16
            n_blocks = thread_configs[1] // 16
            threads = thread_configs[2]

            c_dtype = "half" if dtype == "fp16" else "nv_bfloat16"

            template_str = jinja2.Template(TEMPLATE).render(
                scalar_t=c_dtype,
                w_type_id=scalar_type + ".id()",
                threads=threads,
                thread_m_blocks=max(m_blocks, 1),
                thread_n_blocks=n_blocks,
                thread_k_blocks=k_blocks,
                m_block_size_8=m_blocks == 0.5,
                stages="pipe_stages",
                has_act_order=has_act_order,
                has_zp=has_zp,
                group_blocks=group_blocks,
                is_zp_float=False,
            )

            all_template_str_list.append(template_str)

        file_content = FILE_HEAD + "\n\n"
        file_content += "\n\n".join(all_template_str_list) + "\n\n}\n"
        filename = f"kernel_{dtype}_{scalar_type[8:].lower()}.cu"

        with open(os.path.join(os.path.dirname(__file__), filename), "w") as f:
            f.write(file_content)


if __name__ == "__main__":
    remove_old_kernels()
    generate_new_kernels()