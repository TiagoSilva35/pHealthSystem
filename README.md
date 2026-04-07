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

## Testes

```zsh
python -m unittest discover -s tests -p "test_*.py"
```

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