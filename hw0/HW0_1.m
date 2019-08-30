%close figure windows and clear workspace
close all; clear all; clc; warning off;

%define vectors
v_1 = [sqrt(3), 1, 0].'/2;
v_2 = [0, 1, sqrt(3)].'/2;
vh_1 = [1, 0, -1].'/sqrt(2);
vh_2 = [-1, 1, 0].'/sqrt(2);

%generate complement
v_3 = cross(v_1, v_2) / norm(cross(v_1, v_2));
vh_3 = cross(vh_1, vh_2) / norm(cross(vh_1, vh_2));

%solve
A = [vh_1, vh_2, vh_3] * inv([v_1, v_2, v_3]);

%check
[vh_1, vh_2, vh_3] - A*[v_1, v_2, v_3]
[v_1, v_2, v_3] - inv(A)*[vh_1, vh_2, vh_3]