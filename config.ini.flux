; Set model kwargs here by editing the listed values below:

[model parameters]
; snow optical depth
tau_tot=1000.0

; mass concentration of impurity [mIMP/(mIMP+mICE)]
imp_cnc=0e-9

; snow density (kg/m3, only needed if flg_crt=1)
rho_snw = 200

; ice density (kg/m3)
rho_ice = 917

[plot options]
; plot in optical depth space (=0) or Cartesian space (=1)?
flg_crt=1

; plot in 2-D (=0), 3-D (=1). or no plot (=999)?
flg_3D=999

[data]
; directory to wrtie output data
output_dir=/scratch/climate_flux/amschne

; directory of optics files
optics_dir=/scratch/climate_flux/amschne

; impurity optics file
fi_imp=mie_sot_ChC90_dns_1317.nc