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

int16 weight0[10 * N_FEATURES] = {
    -226, 109, 950, -968, 862, 989, -367, -936, -519, -537,
    861, -7, 435, -424, 858, -965, -900, 904, -806, 1000, 559,
    -460, 814, 436, -248, 671, -403, -738, 409, -786, 527,
    772, -485, -1018, -106, 81, 864, -245, -288, 783
};
int16 bias0[10] = {
    582, -141, -708, -250, -763, 449, -221, -511, -687, 947
};

int16 weight1[N_OUT * 10] = {
    874, -166, -450, 325, -40, 669, 977, -464, 942, 333
};
int16 bias1[N_OUT] = {-937};

int16 xMean[N_FEATURES] = {-318, -514, 942, 333};
int16 xStd[N_FEATURES] = {12, 559, -40, 669};

void MODEL_Init()
{
    DenseDefI16 layers[] = {
        {10, N_FEATURES, ACT_FN_RELU, weight0, bias0},
        {N_OUT, 10, ACT_FN_STRAIGHT, weight1, bias1},
    };
    nn = fcnn_eval_create_i16(layers, 2);
}

extern void TESTSTART_Task1(void);

extern void TESTEND_Task1(void);

void MODEL_Run()
{
    TESTSTART_Task1();
    int16 xRaw[N_FEATURES] = {1, 2, 3, 4};
    int16 xTf[N_FEATURES] = {0};
    for (uint8 i = 0; i < N_FEATURES; i++) {
        xTf[i] = (int16) ((xRaw[i] - xMean[i]) / xStd[i]);
    }
    fcnn_eval_forward_i16(nn, xTf);
    TESTEND_Task1();
}
