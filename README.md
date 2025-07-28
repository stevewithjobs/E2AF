# E2AF

This repository implements **E2AF**: an agent-enhanced framework for online multi-source traffic flow prediction.

## Files

- `train.py`: Train the multi-scale forecasting model.
- `train_rl.py`: Train the decision agent to adaptively select experts or input lengths.
- `test_rl.py`: Evaluate the decision agent on test data using the learned policy.

## Usage

```bash
# Train multi-scale forecasting model
python train.py

# Train decision agent
python train_rl.py
