# train_mnist_tf.py
import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import argparse


def build_and_compile_cnn_model():
    """Build and compile a simple CNN model for MNIST"""
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    return model


def setup_multi_worker_strategy():
    """Setup MultiWorkerMirroredStrategy for distributed training"""
    # Training Operator v1 automatically sets TF_CONFIG
    tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))

    if tf_config:
        print(f"TF_CONFIG: {json.dumps(tf_config, indent=2)}")
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        print(f"Number of devices: {strategy.num_replicas_in_sync}")
    else:
        print("Running in single-worker mode")
        strategy = tf.distribute.get_strategy()

    return strategy


def prepare_dataset(batch_size_per_replica, strategy):
    """Load and prepare MNIST dataset"""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess data
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    # Calculate global batch size
    global_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

    # Create and distribute datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=10000).batch(global_batch_size)
    train_dataset = train_dataset.repeat()

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(global_batch_size)

    # Distribute datasets across workers
    train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    test_dataset = strategy.experimental_distribute_dataset(test_dataset)

    return train_dataset, test_dataset, len(x_train), len(x_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--checkpoint-dir', type=str, default='/workspace/checkpoints')
    parser.add_argument('--log-dir', type=str, default='/workspace/logs')
    args = parser.parse_args()

    # Setup distributed strategy
    strategy = setup_multi_worker_strategy()

    # Prepare dataset
    train_dataset, test_dataset, train_size, test_size = prepare_dataset(
        args.batch_size, strategy
    )

    steps_per_epoch = train_size // (args.batch_size * strategy.num_replicas_in_sync)
    validation_steps = test_size // (args.batch_size * strategy.num_replicas_in_sync)

    # Build model within strategy scope - THIS IS CRITICAL
    with strategy.scope():
        model = build_and_compile_cnn_model()

    # Callbacks - only on chief worker
    callbacks = []
    tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))
    task_type = tf_config.get('task', {}).get('type', 'chief')
    task_index = tf_config.get('task', {}).get('index', 0)

    is_chief = (task_type == 'chief') or (task_type == 'worker' and task_index == 0)

    if is_chief:
        # Only chief saves checkpoints and logs
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.checkpoint_dir, 'model_epoch_{epoch:02d}.h5'),
            save_freq='epoch',
            save_best_only=True,
            monitor='val_sparse_categorical_accuracy',
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint_callback)

        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=args.log_dir,
            histogram_freq=1,
            update_freq='epoch'
        )
        callbacks.append(tensorboard_callback)

        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_sparse_categorical_accuracy',
            patience=3,
            mode='max',
            restore_best_weights=True
        )
        callbacks.append(early_stopping)

    # Train model
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Global batch size: {args.batch_size * strategy.num_replicas_in_sync}")

    history = model.fit(
        train_dataset,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1 if is_chief else 0  # Only chief prints progress
    )

    # Final evaluation and saving on chief worker only
    if is_chief:
        test_loss, test_accuracy = model.evaluate(test_dataset, steps=validation_steps)
        print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Save in both formats
        model.save(os.path.join(args.checkpoint_dir, 'final_model.h5'))
        model.save(os.path.join(args.checkpoint_dir, 'saved_model'), save_format='tf')
        print("Models saved successfully")


if __name__ == '__main__':
    main()
