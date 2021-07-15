[TFG de Òscar Lorente Corominas](https://upcommons.upc.edu/handle/2117/329577?show=full).

## Sistema

El sistema desarrollado consta de 3 partes principales:

* Detección de peatones (u otros objetos) en imágenes RGB usando YOLOv3
* "Transferencia" de estas etiquetas a las nubes de puntos 3D (usando matrices de proyección) para generar clusters de peatones en 3D (también llamados frustums)
* Preparación de los frustums para entrenar/usar el clasificador [PointNet2](https://github.com/oscar-lorente/pointnet2)

## Código

### Código fuente (src)

En `./src` hay un subdirectorio correspondiente a cada parte principal, además de `./src/utils`. 

#### Src/utils

Dentro de `./src/utils` hay varios ficheros útiles para gestionar las bounding boxes (BBs), histogramas (por si queremos crear un histograma de estadísticas de las BBs, que se pueden obtener con `get_boxes_stats.cpp`), y lo que llamamos "mapas". Estos mapas se usan para almacenar y gestionar ficheros de manera más comprimida. Hay 3 tipos de mapas: "gt", "results" y "matching". Los dos primeros almacenan (serializan) las anotaciones o las predicciones de las BBs de las imágenes de cada escena. Por ejemplo, los resultados de YOLO de todas las imágenes de la escena 1 se almacenan en "scene1_964_results_map.txt" (964: número de BBs detectadas en toda la escena). Los mapas "matching" guardan las correspondencias entre los ficheros de imágenes RGB y las nubes de puntos. Aunque una imagen se corresponda con una nube de puntos, los nombres de estos ficheros no son iguales, ya que este nombre incluye el timestamp de la captura y hay una diferencia de milisegundos. Por ende, en los mapas (por ejemplo, "scene2_matching_map.txt" para las imágenes/nubes de la escena 2) se almacenan las correspondencias. El fichero `generate_maps.cpp` se usa para generar estos mapas.

#### Src/image_pedestrian_detector

Dentro de `./src/image_pedestrian_detector` hay una función principal: `image_detect_pedestrians.cpp` usada para detectar peatones (u otros objetos, aunque habría que cambiar algunas partes del código) en las imágenes indicadas al llamar a la función. El código completo para usar YOLO está en `image_pedestrian_detection.cpp`. Para evaluar las detecciones, se usa el fichero `evaluate_results.cpp` (que hace uso de `evaluation.cpp` y `get_precision_recall.cpp`. Esta evaluación necesita etiquetas (groundtruth), que se ha creado usando [Yolo Mark](https://github.com/oscar-lorente/Yolo_mark). Finalmente, `generate_non_pedestrian_boxes.cpp` se usa para generar BBs de la clase "negativa", es decir, de "no peatones". Para ello se usan las detecciones de peatones, y se exige que los "no peatones" no estén solapados con los peatones. Estos se usan para entrenar PointNet2.

#### Src/labeling_transfer

Dentro de `./src/labeling_transfer` hay una función principal: `transfer_labels.cpp` usada para transferir las etiquetas de las imágenes 2D (BBs) a las nubes de puntos 3D usando matrices de proyección. Este se apoya en `labeling_transfer.cpp`, donde se hacen todos los cálculos necesarios para realizar la transferencia con éxito. En la función `getClusters()` se encuentra el código donde se almacenan los clusters 3D en ficheros *.ply*, así como las coordenadas 2D de cada cluster en ficheros *.txt* (útil para otra red neuronal: [Frustum PointNet](https://github.com/oscar-lorente/frustum-pointnets)).

#### Src/point_cloud_pedestrian_detector

Dentro de `./src/point_cloud_pedestrian_detector` hay varias funciones útiles para preparar las nubes de puntos. `display_point_cloud.cpp` se usa para visualizar las nubes de puntos, usando como base `point_cloud_utils`, donde hay otras funcionalidades extra. `organize_ply_hdf5.cpp` se usa para organizar los ficheros *.ply* que luego se usarán para crear ficheros hdf5 (para entrenar y usar PointNet2) con un script (véase el apartado **scripts**). Antes de hacerlo, hay que preprocesar las nubes de puntos usando `preprocess_bbp_point_clouds.cpp` (hay que preprocesar ambos peatones y no peatones), y `check_normalized_point_clouds.cpp` para asegurarse que se ha normalizado correctamente. Finalmente, `bin_to_ply.cpp` es útil para pasar de los ficheros *.bin* con el formato de Beamagine a *.ply* (además de ser muy útil para utilizar scripts de Python, es necesario para usar las redes neuronales de [OpenPCDet](https://github.com/oscar-lorente/OpenPCDet)).

### Librerías (include)

En `./include` hay un directorio correspondiente a cada directorio de `./src`, encargado de gestionar los ficheros *.hpp* necesarios para ejecutar los *.cpp*. En estos, simplemente se encuentra la definición de las clases y funciones utilizadas.

### Scripts

En `./scripts` hay 3 subdirectorios con scripts en python que pueden ser útiles para según qué función.

#### pointnet

Scripts útiles para entrenar/usar PointNet2: diezmar nubes de puntos de peatones y no peatones (`downsample_bbp.py` y `downsample_bbnp.py`), así como una función para crear los ficheros hdf5 (`ply_to_hdf5.py`).

#### frustum

Scripts útiles para usar Frustum PointNet: diezmar nubes de puntos de peatones (`downsample_bbp_frustum.py`), crear pickles necesarios para hacer inferencia con Frustum PointNet (`write_frustums_pickles.py`), eliminar la intensidad de los ficheros *.ply* por si queremos entrenar/hacer inferencia sin usar intensidad (*remove_intensity_frustums.py*), y finalmente organizar una carpeta con los resultados de todas las escenas (e.g. `frustum/results/all`) en diferentes escenas (e.g. `frustum/results/scenes`).

#### plots

Scripts útiles para visualizar curvas de precisión y recall (PointNet2), histogramas, etc.

## Otros ficheros

Finalmente, hay 2 ficheros extra:

* `Neofusion_RGB_25_02_2020_INFO.txt`: contiene las matrices de proyección necesarias para realizar la transferencia de etiquetas de imágenes RGB a nubes de puntos.
* `PCL_Viewer_Parameters.txt`: contiene la configuración de parámetros del visualizador de nubes de puntos (*PCLVisualizer*), para que cada vez que se use la función `./src/point_cloud_pedestrian_detector/display_point_cloud.cpp` no se tenga que ajustar el visualizador.

## Cosas importantes a tener en cuenta

Las nubes de puntos de Beamagine se almacenan en ficheros binarios (*.bin*) con un formato específico, y las **medidas métricas (coordenada x,y,z en el mundo) están en milímetros**. Por ende, cuando se "cargan" las nubes de puntos, hay que dividir los valores entre 1000 (para pasar a metros). Cuando se generan los *.ply* a partir de los *.bin*, hay que asegurarse que las medidas son las que tocan. Es fácil hacerlo con los ficheros *.ply*, pues se pueden visualizar como un fichero de texto normal. Es importante tener en cuenta esto porque puede ser que tras almacenar una nube de puntos en un *.ply* dividiendo los valores entre 1000 (para tenerla en metros), luego se haga un diezmado y se vuelva a dividir entre 1000 al almacenarla de nuevo, teniendo así las medidas en kilómetros.

## Consultas

Para cualquier otra consula, póngase en contacto con [oscar.lorente.co@gmail.com](mailto:oscar.lorente.co@gmail.com).