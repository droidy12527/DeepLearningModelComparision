	q???xl@q???xl@!q???xl@	???L?S?????L?S??!???L?S??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$q???xl@?lV}????A46<il@YTt$?????*	??????@2F
Iterator::Model?=yX?5??!??U
5S@)??m4????1?????R@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateg??j+???!Ak?<?A2@)?^)???1o?gْ1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatr??????!ѨbU?@)2??%䃎?1?i???@:Preprocessing2U
Iterator::Model::ParallelMapV2? ?	??!?ّ????)? ?	??1?ّ????:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???B?i??!????+7@)??0?*x?16iA?i??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey?&1?l?!R:?Z???)y?&1?l?1R:?Z???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorǺ???f?!u?$?y??)Ǻ???f?1u?$?y??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaptF??_??!,?MF??2@)-C??6Z?1??Wr???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???L?S??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?lV}?????lV}????!?lV}????      ??!       "      ??!       *      ??!       2	46<il@46<il@!46<il@:      ??!       B      ??!       J	Tt$?????Tt$?????!Tt$?????R      ??!       Z	Tt$?????Tt$?????!Tt$?????JCPU_ONLYY???L?S??b 