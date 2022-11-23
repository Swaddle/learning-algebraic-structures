n:=8;
for j in [1..NrSmallGroups(n)] do 
	G:=SmallGroup(n,j); 
	M:=MultiplicationTable(G);
	nrow:=Size(M);

	# temp:=String(StructureDescription(G));
	file:= Concatenation(String(j),".csv");
	
	str:="";
	for k in [1..nrow] do
		for l in [1..nrow] do
			Append(str,String(M[k][l]));
			if l < nrow then 
				Append(str,",");
			fi; 
		od;
		if k < nrow then 
			Append(str,"\n");
		fi; 
od;
	
	PrintTo(file,str);
od;