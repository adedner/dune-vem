clear all; clc;

nelx = input('nelx: ');
nely = input('nely: ');


Lx = 1.0;
Ly = 1.0;



dx = Lx / nelx;
dy = Ly / nely;

%fid = fopen('D:\Dropbox\MatlabVEM\SavedMeshes\mymesh.msh','w');
fid = fopen('./partitioned-mesh.msh','w');

numel = nelx * nely;

numnodes = (nelx + 1) * (nely + 1);

fprintf(fid,'$MeshFormat\n');
fprintf(fid,'2.2 0 8\n');
fprintf(fid,'$EndMeshFormat\n');
fprintf(fid,'$Nodes\n');
fprintf(fid,'%3d\n',numnodes);

inel = 1;
nd = 1;
a = -1;
b = 1;
epsilon = 0.2;
for ix = 1:nelx+1
    for iy = 1:nely+1
        
        
        
        
        x = (ix-1) * (dx );
        y = (iy-1) * (dy) ;
        
        if (x~=0 && x~=Lx && y ~= 0 && y~= Ly)
            rx = (b-a)*rand(1,1) + a;
            ry = (b-a)*rand(1,1) + a;
            %rx
            
            %dx * rx * epsilon
            %pause
            %clc
            x = x + dx * rx * epsilon;
            y = y + dy * ry * epsilon;
        end
        
        
        
        
        
        fprintf(fid,'%3d %2.8f %2.8f %2.8f \n', nd, x, y, 0);
        %         plot(x,y,'ok'); pause(0.1); hold on;
        nd = nd + 1;
    end
end

fprintf(fid,'$EndNodes\n');
fprintf(fid,'$Elements\n');
fprintf(fid,'%2d\n',numel);
nd1 = 1;
inel = 1;
dum1 = 3;
dum2 = 0;
for ix = 1:nely
    for iy = 1:nelx
        
        nd2 = nd1 + (nely + 1);
        nd3 = nd2 + 1;
        nd4 = nd1 + 1;
        
        %         1 3 0 1 2 3 4
        
        fprintf(fid,'%2d %2d %2d %2d %2d %2d %2d \n', inel, dum1, dum2, nd1,nd2, nd3, nd4);
        nd1 = nd2;
        inel = inel + 1;
    end
    nd1 = ix + 1;
    
end
fprintf(fid,'$EndElements');
fclose('all');
