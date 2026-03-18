import os
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks
from sklearn.model_selection import train_test_split

# Setup production logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [CHIEF-ML-ENGINE] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SoyoDeepLearningEngine:
    """
    Enterprise-grade Deep Learning Engine for Procurement Risk.
    Architecture: Multi-Task Neural Network with Shared Feature Trunk.
    Optimized for CPU-based training by pre-calculating statistics.
    """
    
    def __init__(self, model_path='models/soyo_dl_v1.keras'):
        self.model_path = model_path
        self.model = None
        
        # Feature Schema
        self.NUMERIC_FEATURES = ['tender_budget_kes', 'credit_score', 'supplier_age_at_award_days']
        self.CATEGORICAL_FEATURES = ['company_size', 'category']
        self.TEMPORAL_FEATURES = ['days_delayed', 'cost_overrun_percentage']
        self.RISK_LABELS = ['Low', 'Medium', 'High']
        
        # Hyperparameters
        self.params = {
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 25,
            'embedding_dim': 16,
            'hidden_units': [64, 32],
            'sequence_length': 3, # Number of past projects to remember
            'temporal_dims': 2    # [delay, cost_overrun]
        }

    def _feature_engineering(self, df):
        """Consistent business logic for target labeling."""
        conditions = [
            (df['contract_status'] == 'Terminated') | (df['days_delayed'] > 30) | (df['cost_overrun_percentage'] > 10),
            (df['days_delayed'] > 7) | (df['cost_overrun_percentage'] > 0)
        ]
        df['risk_level'] = np.select(conditions, [2, 1], default=0) # 0:Low, 1:Medium, 2:High
        return df

    def _create_dataset(self, df, training=True):
        """Optimized tf.data pipeline with Sequence support."""
        # 1. Static Features
        features = {col: df[col].values for col in self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES}
        
        # 2. Temporal Features (Simulating sequences for demonstration)
        # In production, this would be a 3D array (samples, time, features)
        sequences = np.random.randn(len(df), self.params['sequence_length'], self.params['temporal_dims']).astype('float32')
        features['history_input'] = sequences

        targets = {
            'risk_output': df['risk_level'].values,
            'delay_output': df['days_delayed'].values.astype('float32')
        }
        
        ds = tf.data.Dataset.from_tensor_slices((features, targets))
        if training:
            ds = ds.shuffle(len(df)).repeat()
        return ds.batch(self.params['batch_size']).prefetch(tf.data.AUTOTUNE)

    def build_and_compile(self, train_df):
        """Constructs the MTL model with a Hybrid LSTM-Dense Architecture."""
        inputs = {}
        processed_features = []

        # --- BRANCH A: Static Context (Numbers & Categories) ---
        for col in self.NUMERIC_FEATURES:
            inp = layers.Input(shape=(1,), name=col, dtype='float32')
            mean = train_df[col].mean()
            variance = train_df[col].var()
            norm = layers.Normalization(name=f"norm_{col}", axis=None, mean=mean, variance=variance)
            processed_features.append(layers.Reshape((1,))(norm(inp)))
            inputs[col] = inp

        for col in self.CATEGORICAL_FEATURES:
            inp = layers.Input(shape=(1,), name=col, dtype='string')
            vocab = train_df[col].unique().tolist()
            lookup = layers.StringLookup(output_mode='int', name=f"lookup_{col}", vocabulary=vocab)
            emb = layers.Embedding(input_dim=len(vocab) + 1, 
                                   output_dim=self.params['embedding_dim'])(lookup(inp))
            processed_features.append(layers.Reshape((self.params['embedding_dim'],))(emb))
            inputs[col] = inp

        static_trunk = layers.Concatenate()(processed_features)

        # --- BRANCH B: Temporal Context (LSTM Memory) ---
        history_input = layers.Input(shape=(self.params['sequence_length'], self.params['temporal_dims']), 
                                    name='history_input')
        inputs['history_input'] = history_input
        
        # LSTM layer to process performance trends
        temporal_trunk = layers.LSTM(16, dropout=0.2, recurrent_dropout=0.2)(history_input)

        # --- FUSION LAYER ---
        combined = layers.Concatenate()([static_trunk, temporal_trunk])
        
        x = combined
        for units in self.params['hidden_units']:
            x = layers.Dense(units, activation='swish')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)

        # --- TASK HEADS ---
        risk_head = layers.Dense(3, activation='softmax', name='risk_output')(x)
        delay_head = layers.Dense(1, activation='linear', name='delay_output')(x)

        self.model = Model(inputs=inputs, outputs=[risk_head, delay_head])
        
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.params['learning_rate']),
            loss={'risk_output': 'sparse_categorical_crossentropy', 'delay_output': 'huber'},
            loss_weights={'risk_output': 1.0, 'delay_output': 0.1},
            metrics={'risk_output': 'accuracy', 'delay_output': 'mae'}
        )
        logger.info("Hybrid LSTM-MTL Architecture Compiled.")

    def train(self, csv_path):
        """Full training lifecycle."""
        df = pd.read_csv(csv_path)
        df = self._feature_engineering(df)
        
        train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
        
        train_ds = self._create_dataset(train_df)
        val_ds = self._create_dataset(val_df, training=False)
        
        self.build_and_compile(train_df) 
        
        stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        logger.info("Starting Hybrid Training Cycle...")
        self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.params['epochs'],
            steps_per_epoch=len(train_df) // self.params['batch_size'],
            callbacks=[stop_early],
            verbose=1
        )
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        logger.info(f"Hybrid LSTM Model serialized to {self.model_path}")

    def load(self):
        """Production model loader."""
        if os.path.exists(self.model_path):
            # Custom objects usually needed for complex architectures, but here we use standard layers
            self.model = tf.keras.models.load_model(self.model_path)
            return True
        return False

    def predict(self, request_dict, history=None):
        """Atomic inference for real-time risk scoring."""
        if not self.model:
            if not self.load(): return None
            
        # 1. Prepare Static Inputs
        tensor_inputs = {k: tf.constant([[v]], dtype=tf.float32 if k in self.NUMERIC_FEATURES else tf.string) 
                         for k, v in request_dict.items() if k != 'history'}
        
        # 2. Prepare Temporal Inputs (History)
        if history is None:
            # Fallback: Zero-padding if no history available
            history = np.zeros((1, self.params['sequence_length'], self.params['temporal_dims']), dtype='float32')
        else:
            history = np.array([history], dtype='float32')
            
        tensor_inputs['history_input'] = tf.constant(history)
        
        risk_probs, delay_days = self.model(tensor_inputs, training=False)
        
        risk_idx = np.argmax(risk_probs[0])
        return {
            "risk_level": self.RISK_LABELS[risk_idx],
            "risk_score_probability": float(np.max(risk_probs[0])),
            "predicted_delay_days": float(delay_days[0][0])
        }

if __name__ == "__main__":
    engine = SoyoDeepLearningEngine()
    engine.train('data/procurement_master_dataset.csv')
