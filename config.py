import os

# BASE_PATH = 'R:\\LNLS\\Grupos\\GAMS\\2_Projetos\\22_Monitoramento\\Blindagem\\Raio_Parede\\dados_e_resultados\\historico\\pilar_central'
BASE_PATH = os.path.join(os.path.dirname(os.getcwd()),'dados_e_resultados', 'historico', 'pilar_central')

EXPECTED_POINT_NAMES = {
    'external': ['RC1_Exter_P1', 'RC2_Exter_P1', 'RC3_Exter_P1', 'RC4_Exter_P1', 'RC5_Exter_P1'],
    'internal': ['RC1', 'RC2', 'RC3', 'RC4', 'RC5'],
    'magnets': ['RC1_Ima', 'RC2_Ima', 'RC3_Ima', 'RC4_Ima', 'RC5_Ima']
}

