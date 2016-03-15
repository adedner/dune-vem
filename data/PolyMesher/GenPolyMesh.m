%
% Generate polygonal mesh and the underlying subtriangulation. 
% Write the mesh in GMsh format in METIS format


function aa = GenPolyMesh (NElem)

Domain = @UnitSqDomain ;

MaxIter = 100;

%fid_pmesh = fopen ('/home/gcd3/codes/dune-vem/data/partitioned-mesh.msh','w');

fid_pmesh = fopen ('/home/gcd3/codes/dune-vem/data/partitioned-mesh.msh','w');

[vertices,elements,Supp,Load,P] = PolyMesher(Domain,NElem,MaxIter);


mesh.vertices = vertices;
mesh.elements = elements;

%
%
%
numNodes = size(mesh.vertices,1) ; 
fprintf (fid_pmesh, '$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n') ;
fprintf (fid_pmesh, '%4d\n', numNodes + size(mesh.elements,1) );
for j=1:numNodes
    fprintf (fid_pmesh, '%3d\t%2.16f\t%2.16f\t%2.16f\n',j, mesh.vertices (j,1),...
        mesh.vertices (j,2), 0.0 );
end
% 
 InteriorVertexNumber = numNodes + 1;
 iSubElement = 1;
% hold on;
% % 87 2 2 0 16 37 121 71
 for meshElementInd = 1:numel(mesh.elements) % Loop over all polygons in the mesh
 	globalVertexIDs = mesh.elements{meshElementInd};
    clear Polyvertices;
    Polyvertices(1:length(globalVertexIDs),1) = mesh.vertices (globalVertexIDs,1);
    Polyvertices(1:length(globalVertexIDs),2) = mesh.vertices (globalVertexIDs,2);

     xe = barycentreOfPolygon(Polyvertices);
    % plot(Polyvertices(:,1),Polyvertices(:,2), 'or'); 
    % plot(xe(1),xe(2), '+r'); 
     fprintf (fid_pmesh, '%3d\t%2.16f\t%2.16f\t%2.16f\n',InteriorVertexNumber, xe(1),xe(2),0.0);
     for iVertex = 1:length(globalVertexIDs)
        nd1 = globalVertexIDs (iVertex);
        if (iVertex == length(globalVertexIDs))
        nd2 = globalVertexIDs (1);
        else
            nd2 = globalVertexIDs (iVertex + 1);
        end        
        kdisT = [nd1 nd2 InteriorVertexNumber];
        kod (iSubElement,:) = [iSubElement, 2,4,0,16,1,meshElementInd,kdisT] ; 
         iSubElement = iSubElement + 1;
     end
     InteriorVertexNumber = InteriorVertexNumber + 1;
     %pause;
end
%     
% 
% 
fprintf (fid_pmesh, '$EndNodes\n$Elements\n');
%      
fprintf (fid_pmesh, '%d\n', size(kod,1));
for j=1:size(kod,1)
    fprintf (fid_pmesh, '%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n', kod(j,:) );
end
fprintf (fid_pmesh, '$EndElements');
fclose (fid_pmesh);
fprintf ('mesh file written \n');
end

%%%%%%%%%%%%%%%%%%%
% 	

