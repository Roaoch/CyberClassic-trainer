import warnings
import logging

from src.gan import GAN

warnings.simplefilter("ignore", UserWarning)
logger = logging.getLogger(__name__)

text_generator = GAN(
    min_length=30,
    max_length=50,
    df_path='./dataset.csv',
    false_df_path='./false_dataset.csv',
    is_train_generator=False,
    is_train_discriminator=False,
    is_train_gan=True,
    n_epochs=8
)
text_generator.save()

text_generator.test_generate()
