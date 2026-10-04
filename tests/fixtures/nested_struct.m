function y = nested_struct()
% Nested struct fields are flattened to a single C field (`a__b`, `a__c`).
s.a.b = 3;
s.a.c = 4;
y = s.a.b + s.a.c;
end
