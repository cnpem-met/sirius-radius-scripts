# Sirius' radius data treatment routine
Script used to process raw 3D data files containing radius measurements taken from
outside and inside of Sirius' radiation shielding. 

### How it works
The applied routine is as follows:
1. Scan all the .txt files located on a predefined directory path *
2. For each valid txt (i.e. the ones that has the expected points listed):  
  	2.1. Read the 5 control points' 3D positions  
  	2.2. Fit a plane with the 5 points and extract its normal vector
	2.3. Project the points onto the plane
	2.4. Create a new coordinate system (CS) with plane's normal vector direction
	2.5. Change the points old base to the new calculated one
	2.6. Fit a circle with the projected points (based on the new CS)
	2.7. Calculate the distance between each of the points and the center of the circle  
	2.8. Calculate the average of the distances, which are main metric for historical tracking
3. Output all the distances and it's average in the form of a excel sheet

### Requirements
- The packages used are listed in the [requirements](requirements.txt) file, and python >= 3.7 is recommended
- The directory structure that the script expects is configurable via the [config](config.py) file, but the default is:  
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
- For the results to be correct, all points must be exported from SA based on the default Sirius CS/Frame, which is the Machine-local
- For a .txt file to be valid, it has to have the following point names:  
	[mode: internal] RC1, RC2, RC3, RC4, RC5  
	[mode: external] RC1_Exter_P1, RC1_Exter_P2, RC1_Exter_P3, RC1_Exter_P4, RC1_Exter_P5  
	[mode: magnet] RC1_Ima, RC2_Ima, RC3_Ima, RC4_Ima, RC5_Ima  
