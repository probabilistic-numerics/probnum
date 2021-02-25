"""Known derivatives of initial values for ODE example problems that are used to test
the methods in diffeq/odefiltsmooth/initialize.py."""


import numpy as np

# Lotka-Volterra parameters responsible for these values:
# params = (0.5, 0.05, 0.5, 0.05)
# initval: [20., 20.]
# t0: 0.
LV_INITS = np.array(
    [
        2.00000000e01,
        -1.00000000e01,
        -5.00000000e00,
        1.75000000e01,
        8.75000000e00,
        -9.06250000e01,
        -4.53125000e01,
        9.83593750e02,
        4.91796875e02,
        -1.83900391e04,
        -9.19501953e03,
        5.27121631e05,
        2.63560815e05,
        -2.14720140e07,
        -1.07360070e07,
        2.00000000e01,
        1.00000000e01,
        -5.00000000e00,
        -1.75000000e01,
        8.75000000e00,
        9.06250000e01,
        -4.53125000e01,
        -9.83593750e02,
        4.91796875e02,
        1.83900391e04,
        -9.19501953e03,
        -5.27121631e05,
        2.63560815e05,
        2.14720140e07,
        -1.07360070e07,
    ]
)

# The usual initial values and parameters for the three-body problem
THREEBODY_INITS = np.array(
    [
        0.9939999999999999946709294817992486059665679931640625,
        0.0,
        0.0,
        -2.001585106379082379390865753521211445331573486328125,
        0.0,
        -2.001585106379082379390865753521211445331573486328125,
        -315.5430234888826712053567300105174012152470262808609555489430375415196001267175,
        0.0,
        -315.5430234888826712053567300105174012152470262808609555489430375415196001267175,
        0.0,
        0.0,
        99972.09449511380974582623407652494237674536945822956356849412439883437128612817,
        0.0,
        99972.09449511380974582623407652494237674536945822956356849412439883437128612817,
        6.390281114012432978693829866143426861527192861569087503897123013405669119845963e07,
        0.0,
        6.390281114012432978693829866143426861527192861569087503897123013405669119845963e07,
        0.0,
        0.0,
        -5.104537695521316959384194278813460762798119492148033782222345167969699208514969e10,
        0.0,
        -5.104537695521316959384194278813460762798119492148033782222345167969699208514969e10,
        -5.718989915866635673742953579223755513260251013224240606810015106258694103361678e13,
        0.0,
        -5.718989915866635673742953579223755513260251013224240606810015106258694103361678e13,
        0.0,
        0.0,
        7.315561441063621135318644202108449833826961384616926386892170458762613581663705e16,
        0.0,
        7.315561441063621135318644202108449833826961384616926386892170458762613581663705e16,
        1.171034721872789800168691106581608116625156325498263537591690807229212643359755e20,
        0.0,
        1.171034721872789800168691106581608116625156325498263537591690807229212643359755e20,
        0.0,
        0.0,
        -2.060304783152864457766016457053004312347844856370942449557112224304453885256121e23,
        0.0,
        -2.060304783152864457766016457053004312347844856370942449557112224304453885256121e23,
        -4.287443879083103146988750929545238409085724761354466323771047346244918232948406e26,
        0.0,
        -4.287443879083103146988750929545238409085724761354466323771047346244918232948406e26,
        0.0,
        0.0,
        9.601981786174386486553049044619090977452789641463027654594822277483844742280914e29,
        0.0,
        9.601981786174386486553049044619090977452789641463027654594822277483844742280914e29,
        2.45921764824840811167266438750049754390612890829237569709019616087539546093443e33,
        0.0,
        2.45921764824840811167266438750049754390612890829237569709019616087539546093443e33,
        0.0,
        0.0,
        -6.68835380913736620822410435219200785043714506466369418247249547659103361618871e36,
        0.0,
        -6.68835380913736620822410435219200785043714506466369418247249547659103361618871e36,
        -2.034138186021457890521425446714091788329533957393534339124615127934059591480461e40,
        0.0,
        -2.034138186021457890521425446714091788329533957393534339124615127934059591480461e40,
        0.0,
        0.0,
        6.50892954321055682909266276268766148490204239070955567967642805425258366069523e43,
        0.0,
        6.50892954321055682909266276268766148490204239070955567967642805425258366069523e43,
        2.292092276850920428783678416298069135189529108299808020118152953566355216581644e47,
        0.0,
    ]
).flatten()
