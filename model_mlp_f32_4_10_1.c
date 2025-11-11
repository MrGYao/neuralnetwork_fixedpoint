/** 
 * @file    model_mlp_f32_2_15_5_1.c
 * @brief
 *
 * @author  guo yao
 * @date    2025/11/10
 * @history
 * |    Date    |  Version  |   Author    |    Description   |
 * |------------| --------  |-------------|------------------|
 *
**/

#include "power_ai/ann/fcnn_eval.h"
#include "power_ai/common/typedef.h"

#define N_FEATURES (4)
#define N_OUT (1)

fcnn_eval *nn;

float32 weight0[10 * N_FEATURES] = {
    -1.1116789f, -1.7535834f, 0.4213934f, -1.1681386f, -0.0047732f, -1.5396132f, 0.1363550f, -0.0979245f, 1.4788944f,
    0.6789647f, 1.0974622f, 1.4501241f, -1.7490090f, -1.1687436f, 0.4194092f, -0.3671121f, -0.5137733f, 0.7112186f,
    0.2058037f, 1.3599889f, 0.1337793f, -0.6429195f, -1.4873053f, -0.7953155f, -2.6707919f, -0.4274723f, -0.7845227f,
    -0.2315241f, 1.6277243f, -1.9910604f, -0.7535515f, -1.4276889f, -1.1775594f, 0.0316752f, 0.9173547f, 2.6723980f,
    -1.8752600f, 0.3469920f, 0.9668160f, 2.3396888f
};
float32 bias0[10] = {
    2.6723980f, -1.8752600f, 0.3469920f, 0.9668160f, 2.3396888f, 0.4139829f, 2.1234487f, 2.1105500f, 2.4278626f,
    0.5816234f
};

float32 weight1[N_OUT * 10] = {
    0.3871998f, -0.8434016f, -0.8937252f, 0.3994360f, 0.5318564f, -0.3004588f, -1.4109221f, -1.3472771f, 0.0321627f,
    1.2798208f
};
float32 bias1[N_OUT] = {0.8582829f};

float32 xMean[N_FEATURES] = {-0.9216532f, -0.2435501f, -1.1116789f, -1.7535834f};
float32 xStd[N_FEATURES] = {0.5050835f, -0.6511153f, 0.5318564f, -0.3004588f};

void MODEL_Init()
{
    DenseDefF32 layers[] = {
        {10, N_FEATURES, ACT_FN_RELU, weight0, bias0},
        {N_OUT, 10, ACT_FN_STRAIGHT, weight1, bias1},
    };
    nn = fcnn_eval_create_f32(layers, 2);
}

extern void TESTSTART_Task1(void);
extern void TESTEND_Task1(void);

void MODEL_Run()
{
    TESTSTART_Task1();
    float32 xRaw[N_FEATURES] = {1, 2, 3, 4};
    float32 xTf[N_FEATURES] = {0};
    for (uint8 i = 0; i < N_FEATURES; i++) {
        xTf[i] = (xRaw[i] - xMean[i]) / xStd[i];
    }
    fcnn_eval_forward_f32(nn, xTf);
    TESTEND_Task1();
}
