	
h"lx&Y@
h"lx&Y@!
h"lx&Y@	?>?????>????!?>????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$
h"lx&Y@?6?[ ??A	??g?Y@Y0L?
F%??*	??????5@2F
Iterator::Model??_vO??!      Y@)?<,Ԛ???1_B{	??P@:Preprocessing2P
Iterator::Model::Prefetchy?&1?|?!B{	?%4@@)y?&1?|?1B{	?%4@@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?>????#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?6?[ ???6?[ ??!?6?[ ??      ??!       "      ??!       *      ??!       2		??g?Y@	??g?Y@!	??g?Y@:      ??!       B      ??!       J	0L?
F%??0L?
F%??!0L?
F%??R      ??!       Z	0L?
F%??0L?
F%??!0L?
F%??JCPU_ONLYY?>????b 