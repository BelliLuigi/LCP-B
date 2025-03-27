# Init
gw_d,gw_m = xz_data/N, xz_model/N
ga_d,ga_m = x_data/N, x_model/N
gb_d,gb_m = z_data/N, z_model/N
gw=np.copy(gw_d - gw_m)
ga=np.copy(ga_d - ga_m)
gb=np.copy(gb_d - gb_m)


## RMSprop
gw2 = beta*gw2+(1-beta)*np.square(gw)
ga2 = beta*ga2+(1-beta)*np.square(ga)
gb2 = beta*gb2+(1-beta)*np.square(gb)
w += l_rate*gw/sqrt(epsilon+gw2)
a += l_rate*ga/sqrt(epsilon+ga2)
b += l_rate*gb/sqrt(epsilon+gb2)


## SGD
# defaulting to the vanilla stochastic gradient ascent (SGD)
w += l_rate*gw
a += l_rate*ga
b += l_rate*gb

### regularization (LASSO)
if gamma>0.:
w -= (gamma*l_rate)*sign(w)
a -= (gamma*l_rate)*sign(a)
b -= (gamma*l_rate)*sign(b)