addpath ('./data/PolyMesher');

nlxvec = [16 32 64 128];

for i=1:1
    nelx = nlxvec(i);
    numElems = nelx * nelx ;   
    GenPolyMesh(numElems)   
    system ('./build-cmake/src/semilinear-test/vem-semilinear')
    %numElems = 2* numElems;
    
end
