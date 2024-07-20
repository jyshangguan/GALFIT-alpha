from components import *
from astropy.io import fits
import subprocess


class Config:
    def __init__(self, input_file, output_file=None, psf_file='none', sigma_file='none', mask_file='none'):
        self.__input__ = StrParam('A', input_file)
        if output_file is None:
            output_file = input_file.replace('.fit', '_out.fits')
        self.__output__ = StrParam('B', output_file)
        self.__psf__ = StrParam('D', psf_file)
        self.__sigma__ = StrParam('C', sigma_file)
        self.__mask__ = StrParam('F', mask_file)
        self.__mode__ = StrParam('P', 0)
        input_file = fits.open(self.__input__.value)
        psf_file = fits.open(self.__psf__.value)
        scale = self.__read_header__(psf_file[0], 'SCALE')
        self.__psf_scale__ = StrParam('E', scale)
        self.__constrains__ = StrParam('G', 'none')
        in_s1 = self.__read_header__(input_file[0], 'NAXIS1')
        in_s2 = self.__read_header__(input_file[0], 'NAXIS2')
        self.__image_region__ = StrParam('H', f"1 {in_s1} 1 {in_s2}")
        psf_s1 = self.__read_header__(psf_file[0], 'NAXIS1')
        psf_s2 = self.__read_header__(psf_file[0], 'NAXIS2')
        self.__convolution_size__ = StrParam('I', f"{psf_s1} {psf_s2}")
        zp = self.__read_header__(input_file[0], 'ZPT_GSC')
        self.__zeropoint__ = StrParam('J', zp)
        cd11 = self.__read_header__(input_file[0], 'CD1_1')
        cd12 = self.__read_header__(input_file[0], 'CD1_2')
        cd21 = self.__read_header__(input_file[0], 'CD2_1')
        cd22 = self.__read_header__(input_file[0], 'CD2_2')
        dx = np.sqrt(cd11**2+cd12**2)
        dy = np.sqrt(cd21**2+cd22**2)
        self.__pixel_scale__ = StrParam('K', f"{dx} {dy}")
        self.__Display_type__ = StrParam('O', 'regular')
        input_file.close()
        psf_file.close()
        self.parameters = [self.__input__, self.__output__, self.__sigma__, self.__psf__,
                           self.__psf_scale__, self.__mask__, self.__constrains__, self.__image_region__,
                           self.__convolution_size__, self.__zeropoint__, self.__pixel_scale__, self.__Display_type__, self.__mode__]

    def __read_header__(self, hdu, key):
        if key in hdu.header:
            return hdu.header[key]
        else:
            return hdu.header['_'+key[1:]]

    @property
    def galfit_mode(self):
        return self.__mode__.value

    @galfit_mode.setter
    def galfit_mode(self, mode):
        self.__mode__.value = mode

    @property
    def pixel_scale(self):
        value = re.split(r'\s+', self.__pixel_scale__.value)
        dx, dy = float(value[0]), float(value[1])
        return np.sqrt(dx**2+dy**2) * 3600

    @property
    def zeropoint(self):
        value = self.__zeropoint__.value
        if isinstance(value, str):
            value = re.split(r'\s+', value)
            return float(value[0])
        return value

    def __repr__(self) -> str:
        s = ''
        for param in self.parameters:
            s += param.__repr__() + '\n'
        return s


class GalfitTask:
    def __init__(self, config):
        self.__config__ = config
        self.__components__ = []

    @property
    def config(self):
        return self.__config__

    @property
    def components(self):
        return self.__components__

    def add_component(self, component: Component):
        self.__components__.append(component)

    def remove_component(self, index=-1):
        self.__components__.pop(index)

    def __repr__(self) -> str:
        s = self.__config__.__repr__() + '\n'
        for component in self.__components__:
            s += component.__repr__() + '\n'
        return s

    def read_component(self, file_name):
        self.__components__ = []
        with open(file_name, 'r') as file:
            line = file.readline()
            while line:
                line = line.lstrip()
                pos = line.find(')')
                if len(line) > 0 and pos > 0:
                    if line[0] == '0':
                        line = line.split(' ')
                        component = component_names[line[1]]()
                        file = component.read(file)
                        self.__components__.append(component)
                line = file.readline()

    def run(self, galfit_file=None, galfit_mode=0):
        if galfit_file is None:
            galfit_file = self.__config__.__output__.value.replace(
                '.fits', '.galfit')
        self.config.galfit_mode = galfit_mode
        with open(galfit_file, 'w') as file:
            print(self, file=file)
        subprocess.run(['galfit', galfit_file], check=True)
