import configparser

config = configparser.ConfigParser()
config.read('../config.ini')
print(config.sections())

for section in config.sections():
    print(section)
    for option in config.options(section):
        print ('\t', option, '\t\t', config[section][option])
        
config['LOCATION']['weightslocation']
