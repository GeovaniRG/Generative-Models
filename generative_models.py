// Geovani Rodriguez //

import tensorflow as tf
import numpy as np

# Define the generator model
def generator_model(input_dim, hidden_dim, seq_length):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(hidden_dim, input_dim=input_dim, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LSTM(hidden_dim, return_sequences=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hidden_dim, activation='softmax')))
    return model

# Define the discriminator model
def discriminator_model(input_dim, hidden_dim, seq_length):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(hidden_dim, input_shape=(seq_length, input_dim), return_sequences=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hidden_dim, activation='relu')))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='sigmoid')))
    return model

# Define the GAN model
def gan_model(generator, discriminator):
    model = tf.keras.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# Compile the models
input_dim = 100
hidden_dim = 128
seq_length = 10
num_epochs = 100
batch_size = 32
num_batches = data.shape[0] // batch_size

generator = generator_model(input_dim, hidden_dim, seq_length)
discriminator = discriminator_model(input_dim, hidden_dim, seq_length)
gan = gan_model(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Train the model
for epoch in range(num_epochs):
    for batch_idx in range(num_batches):
        # Get a batch of real text data
        real_text = data[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        real_labels = np.ones((batch_size, 1))
        
        # Generate a batch of fake text data
        noise = np.random.normal(0, 1, (batch_size, input_dim))
        fake_text = generator.predict(noise)
        fake_labels = np.zeros((batch_size, 1))
        
                # Train the discriminator
        d_loss_real = discriminator.train_on_batch(real_text, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_text, fake_labels)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, input_dim))
        g_loss = gan.train_on_batch(noise, real_labels)

        # Print the loss values
        if batch_idx % 100 == 0:
            print("Epoch: %d, Batch: %d, Discriminator Loss: %f, Generator Loss: %f" % (epoch, batch_idx, d_loss, g_loss))
