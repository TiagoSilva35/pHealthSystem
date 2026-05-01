# pHealthSystem

Estação de paciente com BITalino (PLUX) para aquisição e visualização de sinais biomédicos via Bluetooth.

## Funcionalidades da Fase 2

- Descoberta de dispositivos Bluetooth nas proximidades.
- Seleção interativa de um dispositivo para ligação.
- Leitura contínua dos canais analógicos do BITalino.
- Visualização em tempo real dos sinais.
- Exportação dos sinais para CSV com timestamps.

## Requisitos

- Linux com Bluetooth ativo.
- Python 3.10.17 (recomendado).
- Dependências do sistema para compilar Bluetooth nativo:

```zsh
sudo apt update
sudo apt install -y build-essential libbluetooth-dev
```

## Ambiente virtual

```zsh
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Execução

Toda a configuração da app está em `src/helpers/constants.py` (MAC fallback, scan duration, canais, sampling rate, nsamples, duração e live plot).

Executa o modo normal (scan + seleção + gráfico live):

```zsh
python -m src.main
```

## Ficheiros de saída

- `bitalino_signals.csv`: timestamp + todos os canais analógicos selecionados.
- `ecg_samples.csv`: timestamp + canal ECG (quando `ECG_ANALOG_CHANNEL` estiver nos canais escolhidos).

## Pipeline para avaliar `ecg_samples.csv`

Para correr o fluxo completo (limpeza + deteção de batimentos/extrasístoles + gráficos):

```zsh
python src/run_ecg_csv_pipeline.py --input ecg_samples.csv --output-dir ecg_pipeline_results
```

Também pode usar diretamente o `run_mitdb.py` com chamada opcional para CSV local:

```zsh
python src/run_mitdb.py --ecg-csv ecg_samples.csv --output ecg_pipeline_results
```

Saídas principais:

- `ecg_pipeline_results/extrasystole_peak_times.png`: gráfico final com batimentos e candidatos a extrasístole.
- `ecg_pipeline_results/pvc_features_dashboard.png`: dashboard com RR, largura QRS e candidatos.
- `ecg_pipeline_results/pvc_features.csv`: tabela de batimentos e features por batimento.
- `ecg_pipeline_results/ecg_pipeline_summary.csv`: resumo global (beats detetados e extrasístoles candidatas).

## Baseline MLP para PVC (MIT-BIH)

Gerar dataset supervisionado (features por batimento + labels a partir de matching com anotações):

```zsh
python src/collect_mlp_dataset.py --database mitdb --output mlp_pvc_dataset.npz
```

Treinar MLP e guardar pesos + parâmetros de normalização:

```zsh
python src/train_mlp_pvc.py --dataset mlp_pvc_dataset.npz --model-output mlp_pvc_model.pt --scaler-output mlp_scaler_params.npz
```

Avaliacao com rede neural integrada no pipeline:

```zsh
python src/run_mitdb.py --database mitdb --evaluation-mode pvc --detection-rule mlp --output mitdb_results_mlp
```

## Testes

```zsh
python -m unittest discover -s tests -p "test_*.py"
```

## Analisar base CU Ventricular Tachyarrhythmia (1-35)

O script abaixo corre duas analises na base completa:

- Pan-Tompkins + prematuridade RR para candidatos PVC.
- Sinal pos-processado + largura QRS para candidatos PVC.

Tambem permite escolher uma amostra (`1-35`) para gerar os graficos com:

- picos detetados e candidatos de extrasistole (Pan-Tompkins)
- largura QRS detetada para a amostra

```zsh
python src/analyze_cu_vt_database.py \
	--database cu-ventricular-tachyarrhythmia-database-1.0.0 \
	--output-dir cu_analysis_results \
	--sample 1
```

Saidas principais:

- `cu_analysis_results/cu_database_summary.csv`: contagens por amostra e totais da base.
- `cu_analysis_results/cuXX_pan_candidates.png`: grafico Pan-Tompkins da amostra selecionada.
- `cu_analysis_results/cuXX_qrs_width.png`: grafico de largura QRS da amostra selecionada.

## Troubleshooting

- Erro `bluetooth/bluetooth.h: No such file or directory`:
	- Instalar `libbluetooth-dev` no Linux.
- Erro de import de `bitalino`:
	- Confirmar que a `.venv` está ativa e correr `pip install -r requirements.txt`.
- Erro `AttributeError: module 'socket' has no attribute 'BTPROTO_RFCOMM'`:
	- O Python/OS atual não tem stack Bluetooth RFCOMM disponível.
	- Executar a app num Linux host com Bluetooth nativo (evitar ambientes sem suporte de kernel Bluetooth, como alguns WSL/containers).
	- Confirmar que o serviço Bluetooth está ativo no host.
- Não encontra dispositivos no scan:
	- Confirmar Bluetooth ativo e permissões do utilizador.
	- Definir `MAC_ADDRESS` em `src/helpers/constants.py` para forçar um dispositivo conhecido (fallback quando o scan falha).
