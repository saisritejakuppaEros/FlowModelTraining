Here’s the **detailed TODO list** with **file names** included for unit testing each module, ensuring the tests are written **after each module** is implemented.

---

### ✅ **Unit Testing Setup**

#### **1. Setup Test Environment**

* [ ] **Create a `tests/` folder** for organizing unit tests:

  * [ ] **tests/**: Main folder containing all test files.
  * [ ] **tests/fixtures/**: For mock data and configurations.
* [ ] **Install required testing dependencies**:

  * [ ] **PyTorch**: `pip install torch`
  * [ ] **PyTorch Lightning**: `pip install pytorch-lightning`
  * [ ] **WandB**: `pip install wandb`
  * [ ] **TensorBoard**: `pip install tensorboard`
  * [ ] **unittest2** (or other testing libraries): `pip install unittest2` or `pytest`

---

### ✅ **2. Test Data Loading** (`tests/test_data.py`)

* **Module Written**: After implementing **`data/dataset.py`** and **`data/transforms.py`**.

  * [ ] **Create random dataset class** (`CustomDataset`) using `torch.utils.data.Dataset`.
  * [ ] **Test DataLoader**:

    * [ ] Test that **data is loaded** properly with a **batch size**.
    * [ ] Ensure **data transformations** (e.g., resizing, normalization) are applied correctly.
  * [ ] **Test data augmentation** (if applicable):

    * [ ] Verify that **data augmentation** works correctly on random inputs.

* **Unit Test Tasks**:

  * **File Name**: `tests/test_data.py`
  * [ ] Test **batch size loading**:

    * Pass data through the **DataLoader** and ensure the correct batch size.
  * [ ] Test **transformations** (e.g., resizing, normalization):

    * Verify that images are correctly resized and normalized.
  * [ ] Test **data augmentation**:

    * Ensure that augmentations like flipping, rotating, or random cropping work.

---

### ✅ **3. Test Model Architecture** (`tests/test_model.py`)

* **Module Written**: After implementing **`models/base_model.py`** and **`models/flow_model.py`**.

  * [ ] **Test the forward pass**:

    * Pass random inputs through the model and check if the **output shape** is as expected.
    * Ensure that the **flow model layers** are properly connected and functional.
  * [ ] **Test optimizer**:

    * Verify that the optimizer is set up correctly and has the expected parameter groups.

* **Unit Test Tasks**:

  * **File Name**: `tests/test_model.py`
  * [ ] Test **forward pass**:

    * Pass random data through the model and verify that the output tensor shape is correct.
  * [ ] Test **model layers connection**:

    * Ensure that all model layers are linked properly (flow layers, etc.).
  * [ ] Test **optimizer setup**:

    * Check if the optimizer is created with the correct parameters.

---

### ✅ **4. Test Training Loop** (`tests/test_train.py`)

* **Module Written**: After implementing **`training/train.py`**, **`training/trainer.py`**, and **`training/metrics.py`**.

  * [ ] **Test the training script**:

    * Mock the **`main`** function in the `train.py` script.
    * Ensure that the script **runs without errors** when invoked.
    * Check if the model trains for a **few epochs** without crashing.
  * [ ] **Test checkpoint saving/loading**:

    * Verify that **checkpoints** are saved and restored correctly.

* **Unit Test Tasks**:

  * **File Name**: `tests/test_train.py`
  * [ ] Test **training loop execution**:

    * Ensure that the model **trains correctly** for a few epochs (mock data).
  * [ ] Test **checkpoint saving/loading**:

    * Test saving and loading checkpoints during the training process.

---

### ✅ **5. Test Inference** (`tests/test_inference.py`)

* **Module Written**: After implementing **`inference/infer.py`**.

  * [ ] **Test inference functions**:

    * Load the trained model from checkpoint.
    * Generate predictions on random inputs.
    * Ensure that the generated outputs (images or text) are in the correct format.

* **Unit Test Tasks**:

  * **File Name**: `tests/test_inference.py`
  * [ ] Test if the **trained model** can load properly from checkpoint.
  * [ ] Test that **inference** works as expected on random inputs.
  * [ ] Verify that the **output format** (image/text) is correct.

---

### ✅ **6. Test Logging** (`tests/test_logger.py`)

* **Module Written**: After implementing **`utils/logger.py`**.

  * [ ] **Test logging with WandB**:

    * Mock the **WandB** logger and verify that logs are being sent correctly.
    * Check that metrics like loss and learning rate are logged during training.
  * [ ] **Test logging with TensorBoard**:

    * Ensure that TensorBoard logs are being written correctly.
    * Check if loss, metrics, and images (if logged) appear correctly in TensorBoard.

* **Unit Test Tasks**:

  * **File Name**: `tests/test_logger.py`
  * [ ] Test **WandB logging**:

    * Mock **WandB** and verify that logs are being sent correctly.
  * [ ] Test **TensorBoard logging**:

    * Verify that **TensorBoard** logs are correctly written and can be viewed.

---

### ✅ **7. Test Metrics Calculation** (`tests/test_metrics.py`)

* **Module Written**: After implementing **`training/metrics.py`**.

  * [ ] **Test evaluation metrics**:

    * Use random data to calculate metrics like **FID** or **Inception Score**.
    * Verify that the metrics functions return values within expected ranges.

* **Unit Test Tasks**:

  * **File Name**: `tests/test_metrics.py`
  * [ ] Test **FID calculation**:

    * Verify that FID calculation returns a valid result within the expected range.
  * [ ] Test **Inception score**:

    * Ensure that the Inception score is calculated correctly.

---

### ✅ **8. Test Checkpointing** (`tests/utils/test_checkpoint.py`)

* **Module Written**: After implementing **`utils/checkpoint.py`**.

  * [ ] **Test saving and loading checkpoints**:

    * Save the model’s state dictionary to a checkpoint.
    * Load the checkpoint and verify that the model’s state is restored correctly.
    * Ensure optimizer and scheduler states are saved and loaded.

* **Unit Test Tasks**:

  * **File Name**: `tests/utils/test_checkpoint.py`
  * [ ] Test **checkpoint saving**:

    * Ensure that model weights, optimizer state, and scheduler state are saved properly.
  * [ ] Test **checkpoint loading**:

    * Verify that the saved state can be restored correctly for both model and optimizer.

---

### ✅ **9. Mock Random Data for Testing** (`tests/fixtures/mock_data.py`)

* **Module Written**: After implementing **random dataset generation** for unit testing.

  * [ ] **Create a mock dataset**:

    * Implement a `CustomDataset` class that generates random data (images and labels).
    * Use random tensors for image data and random integers for labels.

* **Unit Test Tasks**:

  * **File Name**: `tests/fixtures/mock_data.py`
  * [ ] Ensure that **random data** (images and labels) are generated correctly for testing purposes.

---

### ✅ **10. Final Testing and Running Tests**

* [ ] **Run all unit tests**:

  * Use `unittest` or `pytest` to ensure all tests pass successfully.
  * [ ] Test the entire pipeline with mock data and confirm everything works correctly.
* [ ] **Check Test Coverage**:

  * Use tools like `coverage.py` to ensure sufficient test coverage for the entire codebase.

#### **Commands to Run Tests**

* **Run all tests using `unittest`**:

  ```bash
  python -m unittest discover tests/
  ```

* **Run tests using `pytest`** (if using `pytest`):

  ```bash
  pytest tests/
  ```

---

This list ensures that you implement **unit tests** after each module, starting from **data loading** and **model architecture** to **training**, **inference**, **logging**, **metrics**, and **checkpointing**.

Would you like any additional help with writing the test functions or setting up any part of the codebase?
