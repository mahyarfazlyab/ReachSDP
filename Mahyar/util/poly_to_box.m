function [x_min,x_max] = poly_to_box(poly)
% convert a polytope to a box

    dim = poly.Dim;
    A = [eye(dim);-eye(dim)];
    if ~isequal(poly.A,A)
        disp('Error: input polyhedron must be a box!')
        return
    end
    x_max = poly.b(1:dim);
    x_min = -poly.b(dim+1:end);
end

