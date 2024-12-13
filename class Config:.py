class Config:
    def __init__(self):
        # Paths
        self.dataset_path = "data/shortjokes.csv"  # Path to the dataset
        self.cleaned_data_path = "data/cleaned_jokes.csv"
        self.train_data_path = "data/train.csv"
        self.val_data_path = "data/val.csv"

        # Model settings
        self.model_name = "TheBloke/Llama-2-8b-chat-hf"  # Hugging Face model name
        self.output_dir = "output/"  # Directory to save the model checkpoints
        self.max_seq_length = 128  # Max sequence length for tokenization

        # Training settings
        self.batch_size = 8  # Training batch size
        self.eval_batch_size = 8  # Evaluation batch size
        self.learning_rate = 5e-5  # Learning rate
        self.num_train_epochs = 3  # Number of epochs
        self.warmup_steps = 500  # Warmup steps for learning rate scheduler
        self.logging_steps = 100  # Steps for logging training metrics
        self.save_steps = 500  # Steps to save checkpoints

        # Generation settings
        self.max_length = 50  # Max length for generated jokes
        self.num_beams = 5  # Beam search width
        self.temperature = 1.0  # Sampling temperature

    def make_dirs(self):
        """Create necessary directories if they don't exist."""
        os.makedirs(os.path.dirname(self.cleaned_data_path), exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
