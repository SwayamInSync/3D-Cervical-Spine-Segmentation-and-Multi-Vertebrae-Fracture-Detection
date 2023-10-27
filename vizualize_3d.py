from collections import defaultdict
import pydicom
import itkwidgets
import os
import numpy as np
from glob import glob
import vtk
from vtk.util import numpy_support


filepaths = list(
    glob("test-data/1.2.826.0.1.3680043.5876/*dcm", recursive=True))


dicom_files = [pydicom.dcmread(filepath) for filepath in filepaths]


series_dict = defaultdict(list)
for dcm in dicom_files:
    series_dict[dcm.SeriesInstanceUID].append(dcm)


for series_uid, slices in series_dict.items():
    slices = sorted(slices, key=lambda dcm: float(dcm.ImagePositionPatient[2]))
    volume = np.stack([s.pixel_array for s in slices], axis=-1)
    series_dict[series_uid] = volume


volume_vtk_array = numpy_support.numpy_to_vtk(
    volume.ravel(), deep=True, array_type=vtk.VTK_SHORT)


image_data = vtk.vtkImageData()
image_data.SetDimensions(volume.shape[2], volume.shape[1], volume.shape[0])
image_data.SetSpacing((1, 1, 1))
image_data.GetPointData().SetScalars(volume_vtk_array)


volumeMapper = vtk.vtkSmartVolumeMapper()
volumeMapper.SetInputData(image_data)


# colorFunc = vtk.vtkColorTransferFunction()
# colorFunc.AddRGBPoint(700, 0.6, 0.6, 0.6)    # Grey for spongy bone
# colorFunc.AddRGBPoint(3000, 1.0, 1.0, 1.0)   # White for dense bone

# opacityFunc = vtk.vtkPiecewiseFunction()
# opacityFunc.AddPoint(-1000, 0.0)   # Transparent for all values below bone
# opacityFunc.AddPoint(699, 0.0)
# opacityFunc.AddPoint(700, 1.0)     # Start showing bone with full opacity
# opacityFunc.AddPoint(3000, 1.0)    # Full opacity for dense bone


colorFunc = vtk.vtkColorTransferFunction()

# Air
colorFunc.AddRGBPoint(-1000, 0.0, 0.0, 0.0)  # Black for air

# Soft tissues (e.g., muscles, organs)
colorFunc.AddRGBPoint(50, 0.8, 0.3, 0.3)     # Reddish for soft tissues

# Spongy bone
colorFunc.AddRGBPoint(700, 0.6, 0.4, 0.1)    # Brownish for spongy bone

# Dense bone
colorFunc.AddRGBPoint(2000, 1.0, 1.0, 0.9)   # Off-white for dense bone

opacityFunc = vtk.vtkPiecewiseFunction()

# Air
opacityFunc.AddPoint(-1000, 0.0)

# Soft tissues (e.g., muscles, organs)
opacityFunc.AddPoint(49, 0.0)

opacityFunc.AddPoint(50, 0.0)
opacityFunc.AddPoint(699, 0.0)  # soft tissues

# Spongy bone
opacityFunc.AddPoint(700, 0.0)

# Dense bone
opacityFunc.AddPoint(2000, 1.0)


volumeProperty = vtk.vtkVolumeProperty()
volumeProperty.SetColor(colorFunc)
volumeProperty.SetScalarOpacity(opacityFunc)
volumeProperty.ShadeOff()  # Turn off shading for better performance


volumeRenderer = vtk.vtkVolume()
volumeRenderer.SetMapper(volumeMapper)
volumeRenderer.SetProperty(volumeProperty)


renderer = vtk.vtkRenderer()
renderWin = vtk.vtkRenderWindow()
renderWin.AddRenderer(renderer)
renderInteractor = vtk.vtkRenderWindowInteractor()
renderInteractor.SetRenderWindow(renderWin)


renderer.AddVolume(volumeRenderer)


renderer.SetBackground(0.1, 0.1, 0.1)
renderWin.SetSize(800, 800)


renderInteractor.Start()
