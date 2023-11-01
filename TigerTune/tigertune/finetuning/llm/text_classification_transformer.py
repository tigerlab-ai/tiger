"""Text Classification Finetuning Engine."""
from typing import Any, Dict, Optional, Union
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import scikitplot as skplt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
import matplotlib.pyplot as plt

from tigertune.finetuning.data_types import BaseLLMFinetuneEngine


class TextClassificationTransformersFinetuneEngine(BaseLLMFinetuneEngine):
    """Text Classification transformers finetune engine.

    Args:
        base_model_id (`str`): Base model ID to finetune. (Default: 'distilbert-base-uncased')
        hyperparameters (`Optional[Dict[str, Union[str, int, float]]]`): 
            A dict of hyperparameters to customize fine-tuning behavior.

            Currently supported hyperparameters:

            * `max_length`: Max number of words to tokenize in a given text. (Default: 128)
            * `epochs`: Number of training epochs, when DistilBert's layers are frozen. This should be less than 20. (Default: 6)
            * `learning_rate`: Learning rate. (Default: 5e-5)
            * `finetuning_epochs`: Number of finetuning epochs, when DistilBert's layers are unfrozen. This should be less than 20. (Default: 2)
            * `batch_size`: Batch size. (Default: 128)
            * `steps_per_epoch`: Num of steps per epoch. (Default: None)
            * `dropout`: DistilBert dropout rate. (Default: 0.2)
            * `attention_dropout`: DistilBert attention dropout rate. (Default: 0.2)
            * `layer_dropout`: Additional layers' dropout rate. (Default: 0.2)
            * `probability_threshold`: Probability threshold for binary classification. (Default: 0.5)
            * `loss_gamma`: Gamma param for focal loss function. (Default: 2.0)
            * `loss_alpha`: Alpha param for focal loss function. (Default: 0.2)
            * `random_state`: Fixed value for pseudo-random generator. (Default: 42)
    """

    def __init__(
        self,
        base_model_id: str = 'distilbert-base-uncased',
        notebook_mode: bool = False,
        hyperparameters: Optional[Dict[str, Union[str, int, float]]] = None,
    ) -> None:
        """Init params."""
        import os
        import random
        from transformers import (
            DistilBertTokenizerFast,
            TFDistilBertModel,
            DistilBertConfig,
        )

        pd.plotting.register_matplotlib_converters()
        if notebook_mode:
            get_ipython().run_line_magic('matplotlib', 'inline')

        # To show full text (not truncated)
        pd.set_option('display.max_colwidth', None)

        self.params = {'max_length': 128,
                       'epochs': 6,
                       'learning_rate': 5e-5,
                       'finetuning_epochs': 2,
                       'batch_size': 128,
                       'steps_per_epoch': None,
                       'dropout': 0.2,
                       'attention_dropout': 0.2,
                       'layer_dropout': 0.2,
                       'probability_threshold': 0.5,
                       'loss_gamma': 2.0,
                       'loss_alpha': 0.2,
                       'random_state': 42
                       }
        if hyperparameters is not None:
            print("wll")
            self.params.update(hyperparameters)

        ################################################################################
        # Ensure reproducibility
        ################################################################################
        # Set `PYTHONHASHSEED` environment variable at a fixed value
        os.environ['PYTHONHASHSEED'] = str(self.params['random_state'])

        # Set `python` built-in pseudo-random generator at a fixed value
        random.seed(self.params['random_state'])

        # Set `numpy` pseudo-random generator at a fixed value
        np.random.seed(self.params['random_state'])

        # Set `tensorflow` pseudo-random generator at a fixed value
        tf.random.set_seed(seed=self.params['random_state'])

        ################################################################################
        # Instantiate Tokenizer
        ################################################################################
        # TODO: support other base models.
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            base_model_id)

        ################################################################################
        # Instantiate Base Model
        ################################################################################
        # The pre-trained DistilBERT transformer model
        config = DistilBertConfig(dropout=self.params['dropout'],
                                  attention_dropout=self.params['attention_dropout'],
                                  output_hidden_states=True)
        # TODO: support other base models.
        self.base_model = TFDistilBertModel.from_pretrained(
            base_model_id, config=config)

        # Freeze all DistilBERT layers to preserve pre-trained weights
        for layer in self.base_model.layers:
            layer.trainable = False

        # Build model
        self.model = self.__build_model(self.base_model)

    def __build_model(self, transformer):
        """Build Model
        Build a model for binary classification task, upon the foundation of the BERT or DistilBERT architecture.
        Code adapted from https://github.com/RayWilliam46/FineTune-DistilBERT/tree/main/notebooks

        Args:
            transformer:  The base transformer model object from Hugging Face (either BERT or DistilBERT)
                          without any additional classification components.

        Returns:
            model:  A fully compiled tf.keras.Model, complete with added classification layers, 
                    built on top of the underlying pre-trained model structure.
        """

        # Define weight initializer with a random seed to ensure reproducibility
        weight_initializer = tf.keras.initializers.GlorotNormal(
            seed=self.params['random_state'])

        input_ids_layer = tf.keras.layers.Input(shape=(self.params['max_length'],),
                                                name='input_ids',
                                                dtype='int32')
        input_attention_layer = tf.keras.layers.Input(shape=(self.params['max_length'],),
                                                      name='input_attention',
                                                      dtype='int32')

        # DistilBERT provides an output in the form of a tuple. The first element, found at index 0,
        # corresponds to the hidden state at the output of the model's final layer.
        # This element is a tf.Tensor with dimensions (batch_size, sequence_length, hidden_size=768).
        last_hidden_state = transformer(
            [input_ids_layer, input_attention_layer])[0]

        # Only the output of the [CLS] token from DistilBERT is relevant to us, and this token is located at index 0.
        # Extracting the [CLS] tokens provides us with 2D data.
        cls_token = last_hidden_state[:, 0, :]

        D1 = tf.keras.layers.Dropout(self.params['layer_dropout'],
                                     seed=self.params['random_state']
                                     )(cls_token)

        X = tf.keras.layers.Dense(256,
                                  activation='relu',
                                  kernel_initializer=weight_initializer,
                                  bias_initializer='zeros'
                                  )(D1)

        D2 = tf.keras.layers.Dropout(self.params['layer_dropout'],
                                     seed=self.params['random_state']
                                     )(X)

        X = tf.keras.layers.Dense(32,
                                  activation='relu',
                                  kernel_initializer=weight_initializer,
                                  bias_initializer='zeros'
                                  )(D2)

        D3 = tf.keras.layers.Dropout(self.params['layer_dropout'],
                                     seed=self.params['random_state']
                                     )(X)

        output = tf.keras.layers.Dense(1,
                                       activation='sigmoid',
                                       kernel_initializer=weight_initializer,  # CONSIDER USING CONSTRAINT
                                       bias_initializer='zeros'
                                       )(D3)

        # Define the model
        model = tf.keras.Model(
            [input_ids_layer, input_attention_layer], output)

        # Compile the model
        model.compile(tf.keras.optimizers.Adam(learning_rate=self.params['learning_rate']),
                      loss=self.__focal_loss(),
                      metrics=['accuracy'])

        return model

    def __focal_loss(self):
        """Computes focal loss

        Code adapted from https://gist.github.com/mkocabas/62dcd2f14ad21f3b25eac2d39ec2cc95
        """

        def focal_loss_fixed(y_true, y_pred):
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred,
                            tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true, 0), y_pred,
                            tf.zeros_like(y_pred))
            return -K.mean(self.params['loss_alpha'] * K.pow(1. - pt_1, self.params['loss_gamma']) * K.log(pt_1)) - K.mean((1 - self.params['loss_alpha']) * K.pow(pt_0, self.params['loss_gamma']) * K.log(1. - pt_0))
        return focal_loss_fixed

    def __batch_encode(self, texts, batch_size=256):
        """Batch encode texts. Returns the text encodings along with their attention masks, 
            ready for input into a pre-trained transformer model.

        Args:
            tokenizer:   A tokenizer object derived from the PreTrainedTokenizer class.
            texts:       A list of strings, with each string representing a text.
            batch_size:  An integer determining the number of texts in each batch.
        Returns:
            input_ids:       A sequence of texts encoded as a tf.Tensor object.
            attention_mask:  The attention mask for the texts, encoded as a tf.Tensor object.
        """
        input_ids = []
        attention_mask = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer.batch_encode_plus(batch,
                                                      max_length=self.params["max_length"],
                                                      padding='longest',  # implements dynamic padding
                                                      truncation=True,
                                                      return_attention_mask=True,
                                                      return_token_type_ids=False
                                                      )
            input_ids.extend(inputs['input_ids'])
            attention_mask.extend(inputs['attention_mask'])

        return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_mask)

    def finetune(self, **train_kwargs: Any) -> None:
        """Finetune model.
        Args:
            training_dataset (`str`): Dataset filename to finetune on.
            validation_dataset (`str`): Dataset filename to finetune on.
            model_output_path (`str`): Path to save model output. 
        """
        ################################################################################
        # Train Weights of Added Layers and Classification Head
        ################################################################################
        input_training = pd.read_csv(
            train_kwargs['training_dataset']+"/input.csv")['comment_text']
        input_validation = pd.read_csv(
            train_kwargs['validation_dataset']+"/input.csv")['comment_text']
        output_training = pd.read_csv(
            train_kwargs['training_dataset']+"/output.csv")['isToxic']
        output_validation = pd.read_csv(
            train_kwargs['validation_dataset']+"/output.csv")['isToxic']

        # Encode input_training
        input_training_ids, input_training_attention = self.__batch_encode(
            input_training.tolist())

        # Encode input_validation
        input_validation_ids, input_validation_attention = self.__batch_encode(
            input_validation.tolist())

        # Define callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          mode='min',
                                                          min_delta=0,
                                                          patience=0,
                                                          restore_best_weights=True)
        steps_per_epoch = len(input_training.index) // 128
        if self.params["steps_per_epoch"] is not None:
            steps_per_epoch = len(input_training.index) // 128
        # Train the model
        self.train_history1 = self.model.fit(
            x=[input_training_ids, input_training_attention],
            y=output_training.to_numpy(),
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            steps_per_epoch=steps_per_epoch,
            validation_data=([input_validation_ids, input_validation_attention],
                             output_validation.to_numpy()),
            callbacks=[early_stopping],
            verbose=2
        )

        ################################################################################
        # Unfreeze DistilBERT weights to enable fine-tuning
        ################################################################################
        for layer in self.base_model.layers:
            layer.trainable = True

        # Lower the learning rate to prevent destruction of pre-trained weights
        # Otherwise, you might get NAN values
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-6)

        # Recompile the model
        self.model.compile(optimizer=optimizer,
                           loss=self.__focal_loss(),
                           metrics=['accuracy'])

        # Train the model
        self.train_history2 = self.model.fit(
            x=[input_training_ids, input_training_attention],
            y=output_training.to_numpy(),
            epochs=self.params['finetuning_epochs'],
            batch_size=self.params['batch_size'],
            steps_per_epoch=steps_per_epoch,
            validation_data=([input_validation_ids, input_validation_attention],
                             output_validation.to_numpy()),
            callbacks=[early_stopping],
            verbose=2
        )

        # Save the model
        if train_kwargs["model_output_path"]:
            tf.saved_model.save(
                self.model, train_kwargs["model_output_path"])

    def evaluate(self, **eval_kwargs: Any) -> None:
        """
        Args:
            eval_dataset (`str`): Dataset filename for evaluation.
            eval_output_path (`str`): Path to save eval output.
        """
        # Load test data
        test = pd.read_csv(
            eval_kwargs["eval_dataset"])
        X_test = test['comment_text']
        y_test = test['isToxic']

        # Encode X_test
        X_test_ids, X_test_attention = self.__batch_encode(X_test.tolist())

        ################################################################################
        # Evaluate
        ################################################################################
        y_pred = self.model.predict([X_test_ids, X_test_attention])
        print(len(y_test))
        y_pred_thresh = np.where(
            y_pred >= self.params['probability_threshold'], 1, 0)

        print(y_pred_thresh)

        # Get evaluation results
        accuracy = accuracy_score(y_test, y_pred_thresh)
        auc_roc = roc_auc_score(y_test, y_pred)

        print('Accuracy:  ', accuracy)
        print('ROC-AUC:   ', auc_roc)

        ################################################################################
        # Plot Loss and Confusion Matrix
        ################################################################################
        # Training history
        history_df1 = pd.DataFrame(self.train_history1.history)
        history_df2 = pd.DataFrame(self.train_history2.history)
        history_df = history_df1.append(history_df2, ignore_index=True)

        # Plot training and validation loss over each epoch
        history_df.loc[:, ['loss', 'val_loss']].plot()
        plt.title(label='Training & Validation Loss Over Time',
                  fontsize=17, pad=19)
        plt.xlabel('Epoch', labelpad=14, fontsize=14)
        plt.ylabel('Loss', labelpad=16, fontsize=14)
        print("Minimum Validation Loss: {:0.3f}".format(
            history_df['val_loss'].min()))

        # Save the figure
        plt.savefig(eval_kwargs['eval_output_path']+'loss.png',
                    dpi=300.0, transparent=True)

        # Plot confusion matrix
        skplt.metrics.plot_confusion_matrix(y_test.to_list(),
                                            y_pred_thresh.tolist(),
                                            figsize=(6, 6),
                                            text_fontsize=14)
        plt.title(label='Test dataset Confusion Matrix', fontsize=20, pad=17)
        plt.xlabel('Predicted Label', labelpad=14)
        plt.ylabel('True Label', labelpad=14)

        # Save the figure
        plt.savefig(eval_kwargs['eval_output_path']+'confusionMatrix.png',
                    dpi=300.0, transparent=True)
