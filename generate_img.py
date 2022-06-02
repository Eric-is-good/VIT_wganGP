from w_gp_gan import WGPGAN, WGAN_GP_Test

model = WGPGAN()

generate = WGAN_GP_Test(model)

generate.test_img()