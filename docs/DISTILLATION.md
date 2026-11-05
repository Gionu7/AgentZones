# Distillation and federated learning for Proxy Doctor

Two small models are involved:

| Model | Size | Role | Trained by |
|---|---|---|---|
| **Student classifier** | about 2.5k params (NumPy) | Decides the failure class | Distillation from a teacher, then federated learning across zones |
| **Small LLM** (optional) | 0.5–1.5 B (Qwen2.5) | Explains and talks to the user | LoRA fine-tune on teacher explanations |

The student ensures the doctor works on any laptop, even without a model server. The small LLM makes the
explanations sound like the teacher's.

## 1. Synthetic failure cases

`proxy_doctor/synth.py` generates observations in exactly the format the real diagnostics return. Each case is
labelled with one of 9 classes:

* `healthy`
* `missing_proxy_env`
* `proxy_unreachable`
* `proxy_auth_required`
* `untrusted_corp_ca`
* `tool_ca_bundle`
* `no_proxy_internal`
* `dns_failure`
* `clock_skew`

The error strings are realistic and come from many tools: pip, npm, git, curl, Java, az, docker and others.
The generator also adds noise: unknown fields, unrelated log lines and 2% label noise.

*Site profiles* skew the class mix to mimic real sites:

* `dev-laptops` see mostly CA problems.
* `build-servers` see mostly proxy auth and env problems.
* `k8s-cluster` sees mostly NO_PROXY and DNS problems.

## 2. Teacher labels, then the student

Generate cases and label them with a teacher. The teacher is a big model on the GPU box (vLLM or Ollama) or Azure OpenAI:

```bash
python -m proxy_doctor.distill.make_dataset --n 3000 --out data/distill.jsonl \
    --teacher-url http://gpu-box:8000/v1 --teacher-model Qwen/Qwen2.5-32B-Instruct
```

Leave out `--teacher-url` to run fully offline. The script then uses the generator's ground truth with label
smoothing.

Next, distill into the student (CPU, a few seconds). The loss is `alpha` times the cross-entropy with the
teacher's soft labels, plus `(1 - alpha)` times the cross-entropy with the hard labels:

```bash
python -m proxy_doctor.distill.train_student --data data/distill.jsonl --out models/student.npz
```

`models/student.npz` in the repo was produced offline, without a teacher, and scores about 98% on the
synthetic benchmark. To use a student, set either of these:

* `agents[].options.model_path: models/student.npz` uses it as a fixed model.
* `PD_INIT_FROM=/path/to/models/student.npz` on the aggregator starts federated rounds from it.

The script also logs how often the teacher disagrees with the generator. This is a useful signal to clean up
either the generator or the teacher prompt.

## 3. Optional: a small LLM via LoRA (GPU box or Azure ML)

```bash
python -m proxy_doctor.distill.make_sft --data data/distill.jsonl --out data/sft.jsonl
```

```bash
pip install -e '.[distill]'
```

```bash
python -m proxy_doctor.distill.finetune_lora --data data/sft.jsonl --base Qwen/Qwen2.5-1.5B-Instruct
```

Or as an Azure ML job:

```bash
az ml job create -f deploy/azureml/finetune-job.yml -g <rg> -w <ws>
```

Then convert to GGUF:

```bash
python llama.cpp/convert_hf_to_gguf.py models/proxy-doctor-1.5b/merged --outtype q8_0 \
    --outfile models/proxy-doctor-1.5b.gguf
```

Use it in a zone (the `models/` folder is mounted as `/local-models`):

```bash
MODEL_NAME=proxy-doctor MODEL_SOURCE=/local-models/proxy-doctor-1.5b.gguf COMPOSE_PROFILES=llm scripts/zone.sh az
```

On a laptop:

```bash
agentzone laptop --model proxy-doctor --model-source models/proxy-doctor-1.5b.gguf
```

To share it through the Azure ML registry, register the GGUF once, and download it once where it is needed
(the zones themselves never talk to Azure ML; see [MODELS.md](MODELS.md)):

```bash
az ml model create --registry-name <registry> --name proxy-doctor --version 1 --path models/proxy-doctor-1.5b.gguf --type custom_model
```

```bash
az ml model download --registry-name <registry> --name proxy-doctor --version 1 --download-path models/
```

## 4. Federated learning across zones

Each zone's `fl-client` trains the student on its *local* cases. Today these are synthetic, generated with the
site's profile (`PD_PROFILE`). In production they would be the doctor's own verified diagnoses
(`PD_CASES_PATH`, JSONL `{"obs", "label"}`). The aggregator averages the weights with FedAvg each round; the
plugin is [tasks/proxy_student.py](../tasks/proxy_student.py) and the protocol is in
[ARCHITECTURE.md](ARCHITECTURE.md). Point a client's `FL_MODEL_OUT` and the doctor's `PD_MODEL_PATH` at the same
file and the doctor switches to the federated model when a run is done.

Typical result from `python scripts/demo.py`, measured on the shared synthetic benchmark, which is used for
evaluation only:

| | benchmark accuracy |
|---|---|
| Each site's own model (local data only) | 25–50% on problems from other sites |
| After 1 FedAvg round (az, b, cn) | about 70% |
| After 3 rounds | **about 96%** |

Each site alone is good at its own problems and bad at everyone else's. Federation fixes that without any site
sharing its incident data.

## Questions for the mentors

* **Distillation:** soft labels from a teacher LLM, versus teacher-generated *rationales* plus SFT of the small
  LLM, versus both. What is the right temperature and alpha for so few classes?
* **FL:** non-IID handling (FedProx, per-site personalisation layers). Secure aggregation, so the aggregator in b never sees a
  single zone's update. How much differential-privacy noise is acceptable?
* **Azure ML:** an MCP server around Azure ML (submit, track and register jobs) so that agents in a zone can
  start the GPU fine-tune themselves.
