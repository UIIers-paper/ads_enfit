import tensorflow as tf
from tensorflow.keras import layers, Model

class DiffusionModel(Model):
    def __init__(self, input_dim, timesteps=100):
        super().__init__()
        self.timesteps = timesteps
        self.encoder = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(input_dim)
        ])
    
    def forward_diffusion(self, x, t):
        noise = tf.random.normal(shape=tf.shape(x))
        return x + 0.1 * noise  
    def call(self, inputs):
        x, t = inputs
        noisy_x = self.forward_diffusion(x, t)
        noise_pred = self.encoder(noisy_x)  
        return noise_pred

class RegressionModel(tf.keras.Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.regressor = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])
    
    def call(self, z):
        return self.regressor(z)

class DiffusionRegressor(tf.keras.Model):
    def __init__(self, diffusion_model, regressor):
        super().__init__()
        self.diffusion = diffusion_model
        self.regressor = regressor
    
    def train_step(self, data):
        x, y = data
        t = tf.random.uniform(shape=(tf.shape(x)[0],), maxval=self.diffusion.timesteps, dtype=tf.int32)
        with tf.GradientTape() as tape:
            noise_pred = self.diffusion([x, t])
            loss = tf.reduce_mean(tf.square(noise_pred - x))  # Loss khuếch tán
        grads = tape.gradient(loss, self.diffusion.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.diffusion.trainable_weights))
        
        # Trích xuất latent z (giả lập)
        z = self.diffusion.encoder(x)
        y_pred = self.regressor(z)
        reg_loss = tf.reduce_mean(tf.square(y_pred - y))
        return {"diffusion_loss": loss, "regression_loss": reg_loss}
    


# Encoder VAE
class VAEEncoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(2 * latent_dim)  # Output: mu, log_var
        ])
    
    def call(self, x):
        outputs = self.encoder(x)
        mu, log_var = tf.split(outputs, 2, axis=-1)
        return mu, log_var
latent_dim = 32  # Kích thước không gian tiềm ẩn
# Reparameterization
def sampling(args):
    mu, log_var = args
    batch_size = tf.shape(mu)[0]
    epsilon = tf.random.normal(shape=(batch_size, latent_dim))
    return mu + tf.exp(0.5 * log_var) * epsilon

# Decoder VAE (không bắt buộc)
class VAEDecoder(tf.keras.Model):
    def __init__(self, input_dim):
        super().__init__()
        self.decoder = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(input_dim)
        ])
    
    def call(self, z):
        return self.decoder(z)

# Mạng VAE + Hồi Quy
class VAERegressor(tf.keras.Model):
    def __init__(self, latent_dim, input_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VAEEncoder(latent_dim)
        self.sampling_layer = layers.Lambda(sampling)
        self.decoder = VAEDecoder(input_dim)
        self.regressor = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])
    
    def call(self, inputs):
        mu, log_var = self.encoder(inputs)
        z = self.sampling_layer((mu, log_var))
        recon = self.decoder(z)
        y_pred = self.regressor(z)
        return recon, y_pred, mu, log_var

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            recon, y_pred, mu, log_var = self(x)
            recon_loss = tf.reduce_mean(tf.square(recon - x))
            kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1)
            reg_loss = tf.reduce_mean(tf.square(y_pred - y))
            total_loss = recon_loss + tf.reduce_mean(kl_loss) + reg_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"total_loss": total_loss, "recon_loss": recon_loss, "reg_loss": reg_loss}
    



# Lớp GCN đơn giản
class GCNLayer(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units
    
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            initializer="glorot_uniform",
            trainable=True
        )
    
    def call(self, inputs, adjacency_matrix):
        # GNN: X' = ReLU(A @ X @ W)
        x = inputs
        x = tf.matmul(adjacency_matrix, x)
        x = tf.matmul(x, self.kernel)
        return tf.nn.relu(x)

# Mạng GNN + Hồi Quy
class GNNRegressor(tf.keras.Model):
    def __init__(self, input_dim, hidden_units):
        super().__init__()
        self.gcn1 = GCNLayer(hidden_units)
        self.gcn2 = GCNLayer(hidden_units)
        self.regressor = tf.keras.Sequential([
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])
    
    def call(self, inputs):
        x, adj = inputs
        x = self.gcn1(x, adj)
        x = self.gcn2(x, adj)
        return self.regressor(tf.expand_dims(x, axis=0))  # Global pooling