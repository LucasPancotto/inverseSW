import tensorflow as tf
tf.keras.backend.set_floatx('float32')
import numpy as np



@tf.function
def SWHD(model, coords, params, separate_terms=False):
    """ SWHD equations """

    g    = params[0]
    h0 = params[1]
    lm = params[2]
    gamma = 1
    # scaling factor for hb pred
    #invparam = params[2]

    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch(coords) # coords t , x , y

        out = model(coords)

        Yp = out[0]
        inv = out[1:]

        u  = Yp[:,0] # ux
        h  = Yp[:,1] + h0 # h - h0 + h0

        hb  = inv[0][:,0]

        px = u * (h - hb)

    # First derivatives
    grad_u = tape1.gradient(u, coords)
    u_t = grad_u[:,0]
    u_x = grad_u[:,1]
    # u_y = grad_u[:,2]

    # grad_v = tape1.gradient(v, coords)
    # v_t = grad_v[:,0]
    #v_x = grad_v[:,1]
    # v_y = grad_v[:,2]

    grad_h = tape1.gradient(h, coords)
    h_t = grad_h[:,0]
    h_x = grad_h[:,1]
    # h_y = grad_h[:,2]

    grad_hb = tape1.gradient(hb, coords)
    # hb_t = grad_hb[:,0]
    hb_x = grad_hb[:,1]
    # hb_y = grad_hb[:,2]

    grad_px = tape1.gradient(px , coords)
    div_px = grad_px[: , 1]

    # grad_py = tape1.gradient(py , coords)
    # div_py = grad_py[: , 2]

    del tape1


    # Equations to be enforced
    if not separate_terms:
        f0 = lm*(u_t + u*u_x + g*h_x)
        f1 = h_t + div_px
        return [f0, f1]
    if separate_terms:
        f2 = h_t + div_px  + div_py

        t1 = h_t
        t2 = u_x * (h - hb )
        t3 = u * (h_x - hb_x)
        t4 = v_y * (h - hb )
        t5 = v * (h_y - hb_y )

        return [[f2], [t1, t2, t3 , t4 , t5 ,  u , u_x, v , v_y , h , h_x , h_y , h_t , hb , hb_x , hb_y  ] ]
