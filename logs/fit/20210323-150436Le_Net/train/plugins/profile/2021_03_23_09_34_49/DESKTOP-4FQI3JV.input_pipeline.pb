	?-??{~@?-??{~@!?-??{~@	??t?Uj????t?Uj??!??t?Uj??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?-??{~@?&?W??A?:pΈm~@Yj?q?????*	33333#?@2F
Iterator::Model?5^?I??!6?t?[vV@)i o????14?-??3V@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate????????!?'.y?@)|??Pk???1ڡ??@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?&1???!dp>?c@)䃞ͪϕ?1`???~_@:Preprocessing2U
Iterator::Model::ParallelMapV2?5?;Nс?!?@Б????)?5?;Nс?1?@Б????:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?.n????!M?Yl!M$@)ŏ1w-!?1?w???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?J?4q?!PV=???)?J?4q?1PV=???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?q????o?!???(????)?q????o?1???(????:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??ݓ????!?????@)??H?}]?1???????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??t?Uj??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?&?W???&?W??!?&?W??      ??!       "      ??!       *      ??!       2	?:pΈm~@?:pΈm~@!?:pΈm~@:      ??!       B      ??!       J	j?q?????j?q?????!j?q?????R      ??!       Z	j?q?????j?q?????!j?q?????JCPU_ONLYY??t?Uj??b 