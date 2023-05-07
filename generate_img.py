from train_for_wgangp import WGPGAN, WGAN_GP_Test

model = WGPGAN(model_name="cov")

generate = WGAN_GP_Test(model, model_path="model/cov.pt")

generate.test_img()