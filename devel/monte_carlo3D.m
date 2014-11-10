%function dout = monte_carlo3D(n_photon,wvl,rds_snw)

if (1==1)
    clear;

    %%%%%%%%%%%%%%%   USER INPUT   %%%%%%%%%%%%%%%
    % set number of photons:
    n_photon = 100;

    % wavelength [um]:
    %wvl = 1.3;
    %wvl = 1.55;
    wvl = 0.5;

    % snow effective grain size [um]:
    rds_snw = 100;

end;

% snow optical depth:
tau_tot = 1000.0;

% mass concentration of impurity [mIMP/(mIMP+mICE)]
imp_cnc = 0E-9;

% plot in optical depth space (=0) or Cartesian space (=1)?
flg_crt = 1;

% plot in 2-D (=0), 3-D (=1). or no plot (=999)?
flg_3D = 0;

% snow density (kg/m3, only needed if flg_crt=1)
rho_snw = 200;

% directory of optics files:
[foo host] = system('echo $HOST');
if ((host(1:3)=='flu')|(host(1:3)=='nyx'))
    dir = '/nobackup/flanner/mie/snicar/';
else
    dir = '/data/flanner/mie/snicar/';
end;

% specification for nadir looking sensor:
rsensor = 0.05; % sensor radius [m]
hsensor = 0.1;   % sensor height above snow

%%%%%%%%%%%%   END USER INPUT   %%%%%%%%%%%%%

% retrieve snow and impurity optical properties from NetCDF files, based
% on user-specified wavelength, snow grain size, and impurity
% optics file, specified below

% snow optics:
RRRR             = sprintf('%04d',rds_snw);
fi               = strcat(dir,'ice_wrn_',RRRR,'.nc');
wvl_in           = ncread(fi,'wvl');
ssa_in           = ncread(fi,'ss_alb');
ext_in           = ncread(fi,'ext_cff_mss');
asm_in           = ncread(fi,'asm_prm');
[foo idx_wvl]    = min(abs(wvl*1E-6 - wvl_in));
ssa_ice          = ssa_in(idx_wvl);
ext_cff_mss_ice  = ext_in(idx_wvl);
g                = asm_in(idx_wvl);

% impurity optics
fi_imp           = strcat(dir,'mie_sot_ChC90_dns_1317.nc');
wvl_in_imp       = ncread(fi_imp,'wvl');
ssa_in_imp       = ncread(fi_imp,'ss_alb');
ext_in_imp       = ncread(fi_imp,'ext_cff_mss');
[foo idx_wvl]    = min(abs(wvl*1E-6 - wvl_in_imp));
ssa_imp          = ssa_in_imp(idx_wvl);
ext_cff_mss_imp  = ext_in_imp(idx_wvl);

% test case for comparison with Wang et al (1995) Table 1, and van de
% Hulst (1980). Albedo should be ~0.09739.  Total transmittance
% (diffuse+direct) should be ~0.66096
if (1==0)
    n_photon = 50000;
    tau_tot  = 2.0;
    ssa_ice  = 0.9;
    g        = 0.75;
    imp_cnc  = 0;
end;

if (1==0)
    % manually specify optical properties for test cases and debugging:

    % mass extinction cross-section of ice grains (m2/kg at 500nm)
    ext_cff_mss_ice = 6.6;    % typical for re=250um

    % single-scatter albedo of ice grains (at 500nm):
    ssa_ice = 0.999989859099; % typical for re=250um

    % scattering asymmetry parameter for ice grains:
    g = 0.89;                 % typical for re=250

    % mass extinction cross-section of black carbon (m2/kg at 500nm):
    ext_cff_mss_imp = 12000;

    % single-scatter albedo of black carbon (500nm):
    ssa_imp = 0.30;
end;

% combined mass extinction cross-section of ice+impurity system (m2/kg):
ext_cff_mss = ext_cff_mss_ice*(1-imp_cnc) + ext_cff_mss_imp*imp_cnc;

% calculate probability that extinction event is caused by impurity:
%ams++
% correcting below per Roger and Mark
%P_ext_imp = (imp_cnc*ext_cff_mss_imp)/((1-imp_cnc)*ext_cff_mss_ice);
P_ext_imp = (imp_cnc*ext_cff_mss_imp)/(imp_cnc*ext_cff_mss_imp + (1-imp_cnc)*ext_cff_mss_ice)
%ams--

% cos(theta) array over which to compute function:
costheta_p = [-1.0:0.001:1.0];

% Henyey-Greenstein function:
p = (1-g.^2)./((1+g.^2 - 2.*g.*costheta_p).^(3/2));


% plot phase function versus cos(theta):
if (1==0)
    semilogy(costheta_p,p,'linewidth',3);
    xlabel('cos(theta)','fontsize',18);
    ylabel('Relative Probability','fontsize',18);
    set(gca,'xtick',-1.0:0.2:1.0,'fontsize',16);
end;

%%%%%%%%%%%%%%
% populate PDF of cos(scattering phase angle) with random numbers:
% x (costheta)
if (1==0)
    % old method:    
    r1 = rand(1000000,1);   % distribution from  0 -> 1
    r1 = (r1.*2) - 1;       % distribution from -1 -> 1

    % y (phase function)
    r2 = rand(1000000,1);
    r2 = r2.*max(p);        % distribution from 0 -> max(p)

    % phase function of the random x values:
    y_rand = (1-g.^2)./((1+g.^2 - 2.*g.*r1).^(3/2));

    % include only the random (x,y) pairs that are inside the function:
    idx_inc1 = find(r2 <= y_rand);

    % new array that only contains the "valid" random cos(theta) values:
    p_rand = r1(idx_inc1);

else
    % new method:
    r1 = rand(1000000,1);   % distribution from  0 -> 1
    if(g==0)
        p_rand = 1 - 2.*r1
    else
        p_rand = (1/(2*g)).*(1+g^2-((1-g^2)./(1-g+2*g.*r1)).^2);
    end;
end;    
    
% SANITY CHECK: mean of the random distribution (should equal g)
p_mean = mean(p_rand);


%%%%%%%%%%%%%

% Populate PDF of optical path traversed between scattering events:

% Q1: After an optical path of tau, how many photons have NOT
% suffered an extinction event?

if (1==0)
    % old method
    % x: optical depth between scattering events
    r3 = rand(10000000,1);
    r3 = r3.*max(tau_tot,10);  % distribution from 0 -> max_tau

    % y: relative likelihood of optical path traversed (transmittance)
    r4 = rand(10000000,1);      % distribution from 0 -> 1

    % relative likelihood (transmittance) of the random tau values
    f_rand = exp(-r3);

    % include only the random (x,y) pairs that are inside the function:
    idx_inc2 = find(r4 <= f_rand);
    
    % new array that only contains the "valid" random tau values:
    tau_rand = r3(idx_inc2);
else
    % new method
    tau_rand = -log(rand(1000000,1));
end;

% median of tau_rand should be -log(0.5)=0.6931
tau_median = median(tau_rand);


%%%%%%%%%%%%%
% populate PDF of scattering azimuth angle with random numbers

phi_rand = rand(1000000,1).*2*pi; % distribution from 0 -> 2pi


%%%%%%%%%%%%%
% populate PDF of single-scatter albedo with random numbers

ssa_rand = rand(1000000,1);       % 0 -> 1

%%%%%%%%%%%%%

% populate PDF to determine extinction from ice or impurity:
ext_spc_rand = rand(1000000,1);   % 0 -> 1

%%%%%%%%%%%%%


% counters for saving coordinates of absorption events and exit_top events
i1 = 1;
i2 = 1;
i_sensor  = 0; 

if (flg_3D<2)
    figure;
end;

for n=1:n_photon
    %n
    
    % initialization:
    clear x_tau y_tau z_tau;
    y_tau(1)    = 0;
    x_tau(1)    = 0;
    z_tau(1)    = 0;

    % initial direction cosines
    mux_0       = 0;
    muy_0       = 0;
    muz_0       = -1;
    
    clear x_crt y_crt z_crt;
    y_crt(1)    = 0;
    x_crt(1)    = 0;
    z_crt(1)    = 0;

    path_length = 0;
    
    if (1==0)
        % debugging / demonstration of 2 scattering events:
        
        % 1. photon enters from above, moving straight down:
        i=2;
        dtau_current = 0.2;
        theta_sca    = 0;
        phi_sca      = 50;
        sintheta     = sind(theta_sca);
        costheta     = cosd(theta_sca);
        sinphi       = sind(phi_sca);
        cosphi       = cosd(phi_sca);
        
        if (muz_0 == 1)
            mux_n   = sintheta*cosphi;
            muy_n   = sintheta*sinphi;
            muz_n   = costheta;
        elseif (muz_0 == -1)
            mux_n   = sintheta*cosphi;
            muy_n   = -sintheta*sinphi;
            muz_n   = -costheta;
        else
            mux_n   = (sintheta*(mux_0*muz_0*cosphi - muy_0*sinphi))/(sqrt(1-muz_0^2)) + mux_0*costheta;
            muy_n   = (sintheta*(muy_0*muz_0*cosphi + mux_0*sinphi))/(sqrt(1-muz_0^2)) + muy_0*costheta;
            muz_n   = -sqrt(1-muz_0^2)*sintheta*cosphi + muz_0*costheta;
        end;
        
        % update coordinates:
        x_tau(i)     = x_tau(i-1) + dtau_current*mux_n;
        y_tau(i)     = y_tau(i-1) + dtau_current*muy_n;
        z_tau(i)     = z_tau(i-1) + dtau_current*muz_n;

        % update current direction (0):
        mux_0        = mux_n;
        muy_0        = muy_n;
        muz_0        = muz_n;
        
        
        % 2. photon is scattered in some random direction
        i=3;
        dtau_current = 0.4;
        theta_sca    = 20;
        phi_sca      = 50;
        sintheta     = sind(theta_sca);
        costheta     = cosd(theta_sca);
        sinphi       = sind(phi_sca);
        cosphi       = cosd(phi_sca);
        
        if (muz_0 == 1)
            mux_n   = sintheta*cosphi;
            muy_n   = sintheta*sinphi;
            muz_n   = costheta;
        elseif (muz_0 == -1)
            mux_n   = sintheta*cosphi;
            muy_n   = -sintheta*sinphi;
            muz_n   = -costheta;
        else
            mux_n   = (sintheta*(mux_0*muz_0*cosphi - muy_0*sinphi))/(sqrt(1-muz_0^2)) + mux_0*costheta;
            muy_n   = (sintheta*(muy_0*muz_0*cosphi + mux_0*sinphi))/(sqrt(1-muz_0^2)) + muy_0*costheta;
            muz_n   = -sqrt(1-muz_0^2)*sintheta*cosphi + muz_0*costheta;
        end;
        
        % update coordinates:
        x_tau(i)     = x_tau(i-1) + dtau_current*mux_n;
        y_tau(i)     = y_tau(i-1) + dtau_current*muy_n;
        z_tau(i)     = z_tau(i-1) + dtau_current*muz_n;

        % update current direction (0):
        mux_0        = mux_n;
        muy_0        = muy_n;
        muz_0        = muz_n;
        

        % 2-D plot:
        figure;
        plot(x_tau,z_tau,'linewidth',3);
        axis([-0.14 0.14 -0.5 0]);
        xlabel('Optical Depth (x)','fontsize',18);
        ylabel('Optical Depth (z)','fontsize',18);
        grid on;
        
        % 3-D plot:
        figure;
        plot3(x_tau,y_tau,z_tau,'linewidth',3);
        xlabel('Optical Depth (x)','fontsize',18);
        ylabel('Optical Depth (y)','fontsize',18);
        zlabel('Optical Depth (z)','fontsize',18);

    end; % end debug
    
    
    % scatter the photon inside the cloud/snow until it escapes or is absorbed
    condition = 0;
    i = 1;   
    
    while(condition == 0)
        i = i+1;
        
        % pull random indices from the tau, p, and sign arrays
        % (i.e., shuffle the deck each time so we get different
        % distributions with each photon).  This way, we don't need
        % to re-create the distribution each time.
        idx_p    = randi(length(p_rand),1);
        idx_phi  = randi(length(phi_rand),1);
        idx_tau  = randi(length(tau_rand),1);
        idx_ssa  = randi(length(ssa_rand),1);
        idx_ext  = randi(length(ext_spc_rand),1);

        % distance, in optical depth space, to move photon
        dtau_current = tau_rand(idx_tau);
        
        % scattering phase angle:
        if (i==2)
            % the first photon enters travelling straight down:
            costheta  = 1;
            sintheta  = 0;
        else
            costheta  = p_rand(idx_p);
            sintheta  = sqrt(1 - costheta^2);
        end;
        
        % scattering azimuth angle:
        cosphi       = cos(phi_rand(idx_phi));
        sinphi       = sin(phi_rand(idx_phi));
        
        % new cosine directional angles
        if (muz_0 == 1)
            mux_n   = sintheta*cosphi;
            muy_n   = sintheta*sinphi;
            muz_n   = costheta;
        elseif (muz_0 == -1)
            mux_n   = sintheta*cosphi;
            muy_n   = -sintheta*sinphi;
            muz_n   = -costheta;
        else
            % equations from: http://en.wikipedia.org/wiki/Monte_Carlo_method_for_photon_transport
            mux_n   = (sintheta*(mux_0*muz_0*cosphi - muy_0*sinphi))/(sqrt(1-muz_0^2)) + mux_0*costheta;
            muy_n   = (sintheta*(muy_0*muz_0*cosphi + mux_0*sinphi))/(sqrt(1-muz_0^2)) + muy_0*costheta;
            muz_n   = -sqrt(1-muz_0^2)*sintheta*cosphi + muz_0*costheta;
        end;

        %%%% debug %%%%
        if (1==0)
            % store data for debugging:
            mux_save1(i-1) = mux_0;
            muy_save1(i-1) = muy_0;
            muz_save1(i-1) = muz_0;
            tot_save1(i-1) = mux_0^2 + muy_0^2 + muz_0^2; % must equal 1

            mux_save2(i-1) = mux_n;
            muy_save2(i-1) = muy_n;
            muz_save2(i-1) = muz_n;
            tot_save2(i-1) = mux_n^2 + muy_n^2 + muz_n^2; % must equal 1
       
            theta_save(i-1)    = acos(costheta);
            phi_save(i-1)      = phi_rand(idx_phi);
            costheta_save(i-1) = costheta;
            sintheta_save(i-1) = sintheta;
            cosphi_save(i-1)   = cosphi;
            sinphi_save(i-1)   = sinphi;
        end;
        %%%%%%%%%%%%%%
        
        
        % update tau coordinates:
        x_tau(i)    = x_tau(i-1) + dtau_current*mux_n;
        y_tau(i)    = y_tau(i-1) + dtau_current*muy_n;
        z_tau(i)    = z_tau(i-1) + dtau_current*muz_n;
        
        % update Cartesian coordinates
        x_crt(i)    = x_crt(i-1) + dtau_current*mux_n/(ext_cff_mss*rho_snw);
        y_crt(i)    = y_crt(i-1) + dtau_current*muy_n/(ext_cff_mss*rho_snw);
        z_crt(i)    = z_crt(i-1) + dtau_current*muz_n/(ext_cff_mss*rho_snw);

        % update current direction:
        mux_0        = mux_n;
        muy_0        = muy_n;
        muz_0        = muz_n;
        
        % update path length:
        path_length = path_length + dtau_current/(ext_cff_mss*rho_snw);
        
        
        % was the extinction event caused by ice or impurity?
        if (ext_spc_rand(idx_ext) > P_ext_imp)
            % extinction from ice
            ext_state = 1;
            ssa_event = ssa_ice;
        else
            % extinction from impurity
            ext_state = 2;
            ssa_event = ssa_imp;
        end;
        
        % check for exit status:
        if (z_tau(i) > 0)
            % photon has left the top of the cloud/snow (reflected)
            condition = 1; 
                        
            % extend photon path some distance of through air
            %extend_path = 0.15; % [m]
            %x_crt_exit(1)  = x_crt(i);
            %y_crt_exit(1)  = y_crt(i);
            %z_crt_exit(1)  = z_crt(i);
            %x_crt_exit(2)  = x_crt(i) + extend_path*mux_n;
            %y_crt_exit(2)  = y_crt(i) + extend_path*muy_n;
            %z_crt_exit(2)  = z_crt(i) + extend_path*muz_n;

            % extend photon path to z-plane of nadir-looking sensor:
            extend_path              = (hsensor - z_crt(i))/muz_n; % [m]
            x_crt_exit(1)  = x_crt(i);
            y_crt_exit(1)  = y_crt(i);
            z_crt_exit(1)  = z_crt(i);
            x_crt_exit(2)  = x_crt(i) + extend_path*mux_n;
            y_crt_exit(2)  = y_crt(i) + extend_path*muy_n;
            z_crt_exit(2)  = z_crt(i) + extend_path*muz_n;

            % did the photon intersect the sensor? (i.e., are x and y coordinates
            % of ray-plane intersection point inside of sensor
            % circle?)
            if ((x_crt_exit(2)^2 + y_crt_exit(2)^2) <= rsensor^2)  
                % intersection occured
                i_sensor = i_sensor + 1;
                Delta    = 1;
            else
                Delta    = 0;
            end;            
            
        elseif ((z_tau(i) < -tau_tot) & (i==2))
            % photon has left the bottom of the cloud/snow WITHOUT
            % scattering ONCE (direct transmittance)
            condition = 3;
        elseif (z_tau(i) < -tau_tot)
            % photon has left the bottom of the cloud/snow (diffuse transmittance)
            condition = 2;
        elseif(ssa_rand(idx_ssa) >= ssa_event)
            % photon was absorbed: Archive which species absorbed it:
            if (ext_state==1)
                condition = 4;
            elseif (ext_state==2);
                condition = 5;
            end;
        end;
    
    end
    
    % save the state
    condition_save(n) = condition;
    
    % save the number of scattering events:
    n_scat(n) = i-1;
    
    % save the photon path length:
    path_length_save(n) = path_length;
    
    % plot the photon path
    color_rand = rand(1,3);
    if (flg_crt==0)
        x_plot = x_tau;
        y_plot = y_tau;
        z_plot = z_tau;
    elseif (flg_crt==1)
        x_plot = x_crt;
        y_plot = y_crt;
        z_plot = z_crt;
    end;
        
    if (flg_3D == 0)
        % 2-D plot
        plot(x_plot,z_plot,'color',color_rand);
    elseif (flg_3D == 1)
        % 3-D plot
        plot3(x_plot,y_plot,z_plot,'color',color_rand);
    end;
    
    % mark the final event:
    if (flg_3D<2)
        hold on;
    end;
    
    if (condition == 1)
        if (flg_3D == 0)
            pe = plot(x_plot(end),z_plot(end),'k*');
            set(pe,'markersize',10,'markerfacecolor','r','MarkerEdgeColor','r');
        elseif (flg_3D == 1)
            pe = plot3(x_plot(end),y_plot(end),z_plot(end),'k*');
            set(pe,'markersize',10,'markerfacecolor','r','MarkerEdgeColor','r');
        end;
        
        if (1==1)
            % plot extension through air:
            if (flg_3D == 0)
                pex = plot(x_crt_exit,z_crt_exit,'k--');
                if (Delta == 1)
                    % intersection with sensor
                    pex2 = plot(x_crt_exit(2),z_crt_exit(2),'hr');
                    set(pex2,'markersize',10,'markerfacecolor',[1 0 0],'MarkerEdgeColor',[1 0 0]);
                end;
            elseif (flg_3D == 1)
                pex = plot3(x_crt_exit,y_crt_exit,z_crt_exit,'k--');
                if (Delta == 1)
                    % intersection with sensor
                    pex2 = plot3(x_crt_exit(2),y_crt_exit(2),z_crt_exit(2),'hr');
                    set(pex2,'markersize',10,'markerfacecolor',[1 0 0],'MarkerEdgeColor',[1 0 0]);
                end;
            end;
        end;
    elseif ((condition == 2))
        if (flg_3D == 0)
            pe = plot(x_plot(end),z_plot(end),'k*');
            set(pe,'markersize',10,'markerfacecolor',[0 0.8 0],'MarkerEdgeColor',[0 0.8 0]);
        elseif (flg_3D == 1)
            pe = plot3(x_plot(end),y_plot(end),z_plot(end),'k*');
            set(pe,'markersize',10,'markerfacecolor',[0 0.8 0],'MarkerEdgeColor',[0 0.8 0]);
        end;
    elseif ((condition == 3))
        if (flg_3D == 0)
            pe = plot(x_plot(end),z_plot(end),'kd');
            set(pe,'markersize',15,'markerfacecolor','m','MarkerEdgeColor','m');
        elseif (flg_3D == 1)
            pe = plot3(x_plot(end),y_plot(end),z_plot(end),'kd');
            set(pe,'markersize',15,'markerfacecolor','m','MarkerEdgeColor','m');
        end;
    elseif (condition == 4)
        xabsi(i1) = x_plot(end);
        yabsi(i1) = y_plot(end);
        zabsi(i1) = z_plot(end);
        i1        = i1+1;
        %if (flg_3D == 0)
        %    pe = plot(x_plot(end),z_plot(end),'ks');
        %elseif (flg_3D == 1)
        %    pe = plot3(x_plot(end),y_plot(end),z_plot(end),'ks');
        %end;
        %set(pe,'markersize',15,'markerfacecolor','b','MarkerEdgeColor','b');
    elseif (condition == 5) 
        xabsb(i2) = x_plot(end);
        yabsb(i2) = y_plot(end);
        zabsb(i2) = z_plot(end);
        i2        = i2+1;
        %if (flg_3D == 0)
        %    pe = plot(x_plot(end),z_plot(end),'ks');
        %elseif (flg_3D == 1)
        %    pe = plot3(x_plot(end),y_plot(end),z_plot(end),'ks');
        %end;
        %set(pe,'markersize',15,'markerfacecolor','k','MarkerEdgeColor','k');
    end;
    if (flg_3D<2)
        hold on;
    end;
end

% add line at cloud/snow top:
%xtop = xlim;
%ytop = [0 0];
%plot(xtop,ytop,'k--','linewidth',4);

% add sensor to plot
for a=1:360
    xsens(a) = rsensor*cosd(a);
    ysens(a) = rsensor*sind(a);
end
zsens(1:length(xsens)) = hsensor;
if (flg_3D == 0)
    plot(xsens,zsens,'b','linewidth',1);
    ybounds = ylim; ymin = ybounds(1);
    ylim([ymin 0.11]);
elseif (flg_3D == 1)
    plot3(xsens,ysens,zsens,'b','linewidth',1);
    zbounds = zlim; zmin = zbounds(1);
    zlim([zmin 0.11]);
end;


% plot absorption events:
if (i1 > 1)
    if (flg_3D == 0)
        pei = plot(xabsi,zabsi,'ks');
        set(pei,'markersize',15,'markerfacecolor','b','MarkerEdgeColor','b');
    elseif (flg_3D == 1)
        pei = plot3(xabsi,yabsi,zabsi,'ks');
        set(pei,'markersize',15,'markerfacecolor','b','MarkerEdgeColor','b');
    end;
end;
if (i2 > 1)
    if (flg_3D == 0)
        peb = plot(xabsb,zabsb,'ks');
        set(peb,'markersize',15,'markerfacecolor','k','MarkerEdgeColor','k');
    elseif (flg_3D == 1)
        peb = plot3(xabsb,yabsb,zabsi,'ks');
        set(peb,'markersize',15,'markerfacecolor','k','MarkerEdgeColor','k');
    end;
end;

% plot photon extension paths through air:
%if (i_exittop > 1)
%    for i=1:i_exittop-1
%        if (flg_3D == 0)
%            pex = plot(x_crt_exit(i,:),z_crt_exit(i,:),'k--');
%        elseif (flg_3D == 1)
%            pex = plot3(x_crt_exit(i,:),y_crt_exit(i,:),z_crt_exit(i,:),'k--');
%        end;
%    end
%end;

if (flg_crt==0)
    if (flg_3D == 0)
        xlabel('Optical Depth (x)','fontsize',18);
        ylabel('Optical Depth (z)','fontsize',18);
    elseif (flg_3D == 1)
        xlabel('Optical Depth (x)','fontsize',18);
        ylabel('Optical Depth (y)','fontsize',18);
        zlabel('Optical Depth (z)','fontsize',18);
    end;
elseif (flg_crt==1)
    if (flg_3D == 0)
        xlabel('Distance x [m]','fontsize',18);
        ylabel('Distance z [m]','fontsize',18);
    elseif (flg_3D == 1)
        xlabel('Distance x [m]','fontsize',18);
        ylabel('Distance y [m]','fontsize',18);
        zlabel('Depth z [m]','fontsize',18);
    end;
end;

if (flg_3D < 2)
    grid on;
end;


if (1==0)
    saveas(gcf,'fgr_photonpath1.eps','epsc');
end;

% print number of scattering events simulated for each photon:
n_scat;

% mean number of scatteirng events:
n_scat_mean = mean(n_scat);

% mean path length:
path_length_mean = mean(path_length_save);

% fraction of photons reflected from cloud/snow top:
albedo    = length(find(condition_save==1)) / n_photon;

% fraction of photons transmitted through bottom of cloud/snow (after scattering)
t_diffuse = length(find(condition_save==2)) / n_photon;

% fraction of photons transmitted through bottom of cloud/snow (without scattering)
t_direct  = length(find(condition_save==3)) / n_photon;

% fraction of photons absorbed:
fabs      = length(find(condition_save>=4)) / n_photon;

% write output:
sout1 = strcat('Albedo = ',num2str(albedo));
sout2 = strcat('Direct transmittance = ',num2str(t_direct));
sout3 = strcat('Diffuse transmittance = ',num2str(t_diffuse));
sout4 = strcat('Total transmittance = ',num2str(t_diffuse+t_direct));
sout5 = strcat('Absorptance = ',num2str(fabs));
sout6 = strcat('Mean number of scattering events = ',int2str(n_scat_mean));
sout7 = strcat('Mean path length [m] = ',num2str(path_length_mean));
sout8 = strcat('Fraction of incident photons reaching sensor = ',num2str(i_sensor/n_photon));

if (1==1)
    disp(sout1);
    disp(sout2);
    disp(sout3);
    disp(sout4);
    disp(sout5);
    disp(sout6);
    disp(sout7);
    disp(sout8);
end;

dout(1) = albedo;
dout(2) = i_sensor/n_photon;
