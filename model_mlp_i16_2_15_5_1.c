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

#define N_FEATURES (2)
#define N_OUT (1)

fcnn_eval *nn;

int16 weight0[15 * N_FEATURES] = {
    -226, 109, 950, -968, 862, 989, -367, -936, -519, -537, 861, -7, 435, -424, 858, -965, -900, 904, -806, 1000, 559,
    -460, 814, 436, -248, 671, -403, -738, 409, -786
};
int16 bias0[15] = {
    582,-141,-708,-250,-763,449,-221,-511,-687,947,-106,81,864,-245,-288
};

int16 weight1[5 * 15] = {
    874, -166, -450, 325, -40, 669, 977, -464, 942, 333, 783, 861, 744, 393, 405, 669, -854, 18, -647, 237, -978, 157,
    -814, 1002, -645, -378, -87, 623, 349, 785, 57, -644, 168, 785, -643, -272, -349, 485, 649, 176, -570, 443, 159,
    782, -755, -951, -34, -757, 573, 514, -134, 415, 743, 294, -520, 838, 254, -479, -670, 532, -1000, -714, 57, 783,
    -213, -813, -687, 761, -77, 997, -967, -528, -933, 898, -188
};
int16 bias1[5] = {-937,527,772,-485,-1018};

int16 weight2[N_OUT * 5] = {404, -620, 733, 321, 311};
int16 bias2[N_OUT] = {876};

int16 xMean[N_FEATURES] = {-318,-514};
int16 xStd[N_FEATURES] = {12,559};

void MODEL_Init()
{
    DenseDefI16 layers[] = {
        {15, N_FEATURES, ACT_FN_RELU, weight0, bias0},
        {5, 15, ACT_FN_RELU, weight1, bias1},
        {N_OUT, 5, ACT_FN_STRAIGHT, weight2, bias2}
    };
    nn = fcnn_eval_create_i16(layers, 3);
}

extern void TESTSTART_Task1(void);

extern void TESTEND_Task1(void);

void MODEL_Run()
{
    TESTSTART_Task1();
    int16 xRaw[N_FEATURES] = {1, 2};
    int16 xTf[N_FEATURES] = {0};
    for (uint8 i = 0; i < N_FEATURES; i++) {
        xTf[i] = (int16) ((xRaw[i] - xMean[i]) / xStd[i]);
    }
    fcnn_eval_forward_i16(nn, xTf);
    TESTEND_Task1();
}
