# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import pickle
import asyncio
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import random_uuid


async def generate_python_vllm_output(prompt, llm_engine):
    request_id = random_uuid()
    sampling_parameters = {"temperature": 0, "top_p": 1}
    sampling_params = SamplingParams(**sampling_parameters)

    python_vllm_output = None
    last_output = None

    async for vllm_output in llm_engine.generate(prompt, sampling_params, request_id):
        last_output = vllm_output

    if last_output:
        python_vllm_output = [
            (prompt + output.text).encode("utf-8") for output in last_output.outputs
        ]

    return python_vllm_output

vllm_engine_config = {
    "model": "facebook/opt-125m",
    "gpu_memory_utilization": 0.3,
}
prompts = [
    "The most dangerous animal is",
    "The capital of France is",
    "The future of AI is",
]

llm_engine = AsyncLLMEngine.from_engine_args(
    AsyncEngineArgs(**vllm_engine_config)
)

python_vllm_output = []
for prompt in prompts:
    python_vllm_output.extend(
        asyncio.run(generate_python_vllm_output(prompt, llm_engine))
    )

with open("python_vllm_output.pkl", "wb") as f:
    pickle.dump(python_vllm_output, f)
