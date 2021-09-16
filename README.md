# Sirius' radius data treatment routine
Script used to process raw 3D data files containing radius measurements taken from
outside and inside of Sirius' radiation shielding. 

The applied routine is as follows:
1. Scan all the .txt files located on a predefined directory path *
2. For each valid txt (i.e. the ones that has the expected points listed):  
  2.1. Read the 5 control points' 3D positions  
  2.2. Fit a circle within the 5 points  
	2.3. Project the 5 points onto the plane formed by the circle  
	2.4. Calculate the distance between each projected point and the center of the circle  
	2.5. Calculate the average of the distances  
3. Output all the distances and it's average in the form of a excel sheet

Requirements:
- The correct functioning of the script depends on the following directory structure:  
<pre>
		Raio_Parede (root)
		    |-- scripts (where the .py is)
		    |-- dados_e_resultados
			      |-- historico
				     |-- pilar_central
					      |-- pontos_externos
					      |    |-- txt *
					      |-- pontos_internos
				     		   |-- txt *
</pre>


- For a .txt file to be valid, it has to have the following point names:  
	[mode: internal] RC1, RC2, RC3, RC4, RC5  
	[mode: external] RC1_Exter_P1, RC1_Exter_P2, RC1_Exter_P3, RC1_Exter_P4, RC1_Exter_P5  
