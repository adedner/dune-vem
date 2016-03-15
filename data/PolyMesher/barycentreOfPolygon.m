function c = barycentreOfPolygon(vertices)
% Calculate the barycentre of a polygon specified by its vertices
    vertices(end + 1, :) = vertices(1, :);
    signedAreaComponents = vertices(1:end-1, 1) .* vertices(2:end, 2) -...
                                                vertices(2:end, 1) .* vertices(1:end - 1, 2);
    a = 0.5 * sum(signedAreaComponents);
    c = zeros(1, 2);
    c(1) = 1/(6 * a) * sum((vertices(1:end-1, 1) + vertices(2:end, 1)) .* signedAreaComponents);
    c(2) = 1/(6 * a) * sum((vertices(1:end-1, 2) + vertices(2:end, 2)) .* signedAreaComponents);
end