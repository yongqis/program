from abc import abstractmethod
from functools import partial
import tensorflow as tf
from object_detection.anchor_generators import grid_anchor_generator
from object_detection.core import balanced_positive_negative_sampler as sampler
from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import box_predictor
from object_detection.core import losses
from object_detection.core import model
from object_detection.core import post_processing
from object_detection.core import standard_fields as fields
from object_detection.core import target_assigner
from object_detection.utils import ops
from object_detection.utils import shape_utils

slim = tf.contrib.slim


class FasterRCNNFeatureExtractor(object):
    """Faster R-CNN Feature Extractor definition."""

    def __init__(self,
                 is_training,
                 first_stage_features_stride,
                 batch_norm_trainable=False,
                 reuse_weights=None,
                 weight_decay=0.0):
        """Constructor.

        Args:
          is_training: A boolean indicating whether the training version of the
            computation graph should be constructed.
          first_stage_features_stride: Output stride of extracted RPN feature map.
          batch_norm_trainable: Whether to update batch norm parameters during
            training or not. When training with a relative large batch size
            (e.g. 8), it could be desirable to enable batch norm update.
          reuse_weights: Whether to reuse variables. Default is None.
          weight_decay: float weight decay for feature extractor (default: 0.0).
        """
        self._is_training = is_training
        self._first_stage_features_stride = first_stage_features_stride
        self._train_batch_norm = (batch_norm_trainable and is_training)
        self._reuse_weights = reuse_weights
        self._weight_decay = weight_decay

    @abstractmethod
    def preprocess(self, resized_inputs):
        """Feature-extractor specific preprocessing (minus image resizing)."""
        pass

    def extract_proposal_features(self, preprocessed_inputs, scope):
        """Extracts first stage RPN features.

        This function is responsible for extracting feature maps from preprocessed
        images.  These features are used by the region proposal network (RPN) to
        predict proposals.

        Args:
          preprocessed_inputs: A [batch, height, width, channels] float tensor
            representing a batch of images.
          scope: A scope name.

        Returns:
          rpn_feature_map: A tensor with shape [batch, height, width, depth]
        """
        with tf.variable_scope(scope, values=[preprocessed_inputs]):
            return self._extract_proposal_features(preprocessed_inputs, scope)

    @abstractmethod
    def _extract_proposal_features(self, preprocessed_inputs, scope):
        """
        Extracts features from base net (inception/resnet)
        Extracts first stage RPN features, to be overridden.

        """
        pass

    def extract_box_classifier_features(self, proposal_feature_maps, scope):
        """Extracts second stage box classifier features.

        Args:
          proposal_feature_maps: A 4-D float tensor with shape
            [batch_size * self.max_num_proposals, crop_height, crop_width, depth]
            representing the feature map cropped to each proposal.
          scope: A scope name.

        Returns:
          proposal_classifier_features: A 4-D float tensor with shape
            [batch_size * self.max_num_proposals, height, width, depth]
            representing box classifier features for each proposal.
        """
        with tf.variable_scope(scope, values=[proposal_feature_maps]):
            return self._extract_box_classifier_features(proposal_feature_maps, scope)

    @abstractmethod
    def _extract_box_classifier_features(self, proposal_feature_maps, scope):
        """Extracts second stage box classifier features, to be overridden."""
        pass

    def restore_from_classification_checkpoint_fn(
            self,
            first_stage_feature_extractor_scope,
            second_stage_feature_extractor_scope):
        """Returns a map of variables to load from a foreign checkpoint.

        Args:
          first_stage_feature_extractor_scope: A scope name for the first stage
            feature extractor.
          second_stage_feature_extractor_scope: A scope name for the second stage
            feature extractor.

        Returns:
          A dict mapping variable names (to load from a checkpoint) to variables in
          the model graph.
        """
        variables_to_restore = {}
        for variable in tf.global_variables():
            for scope_name in [first_stage_feature_extractor_scope, second_stage_feature_extractor_scope]:
                if variable.op.name.startswith(scope_name):
                    var_name = variable.op.name.replace(scope_name + '/', '')
                    variables_to_restore[var_name] = variable
        return variables_to_restore


class FasterRCNNMetaArch(model.DetectionModel):
    """Faster R-CNN Meta-architecture definition."""

    def __init__(self,
                 is_training,
                 is_building,
                 num_classes,
                 image_resizer_fn,
                 feature_extractor,

                 first_stage_only,
                 first_stage_anchor_generator,
                 first_stage_atrous_rate,
                 first_stage_box_predictor_arg_scope,
                 first_stage_box_predictor_kernel_size,
                 first_stage_box_predictor_depth,

                 first_stage_minibatch_size,
                 first_stage_positive_balance_fraction,
                 first_stage_nms_score_threshold,
                 first_stage_nms_iou_threshold,
                 first_stage_max_proposals,
                 first_stage_localization_loss_weight,
                 first_stage_objectness_loss_weight,

                 initial_crop_size,
                 maxpool_kernel_size,
                 maxpool_stride,

                 second_stage_mask_rcnn_box_predictor,
                 second_stage_batch_size,
                 second_stage_balance_fraction,
                 second_stage_non_max_suppression_fn,
                 second_stage_score_conversion_fn,
                 second_stage_localization_loss_weight,
                 second_stage_classification_loss_weight,
                 second_stage_classification_loss,
                 second_stage_mask_prediction_loss_weight=1.0,
                 hard_example_miner=None,
                 parallel_iterations=16):
        """FasterRCNNMetaArch Constructor.

        Args:
          is_training: A boolean indicating whether the training version of the
            computation graph should be constructed.
          num_classes: Number of classes.  Note that num_classes *does not*
            include the background category, so if groundtruth labels take values
            in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
            assigned classification targets can range from {0,... K}).
          image_resizer_fn: A callable for image resizing.  This callable
            takes a rank-3 image tensor of shape [height, width, channels]
            (corresponding to a single image) and returns a rank-3 image tensor,
            possibly with new spatial dimensions. See
            builders/image_resizer_builder.py.
          feature_extractor: A FasterRCNNFeatureExtractor object.
          first_stage_only:  Whether to construct only the Region Proposal Network
            (RPN) part of the model.
          first_stage_anchor_generator: An anchor_generator.AnchorGenerator object
            (note that currently we only support
            grid_anchor_generator.GridAnchorGenerator objects)
          first_stage_atrous_rate:A single integer indicating the atrous rate for
            the single convolution op which is applied to the `rpn_features_to_crop`
            tensor to obtain a tensor to be used for box prediction. Some feature
            extractors optionally allow for producing feature maps computed at
            denser resolutions.  The atrous rate is used to compensate for the
            denser feature maps by using an effectively larger receptive field.
            (This should typically be set to 1). (扩展卷积，空洞卷积)
          first_stage_box_predictor_arg_scope: Slim arg_scope for conv2d,
            separable_conv2d and fully_connected ops for the RPN box predictor.
          first_stage_box_predictor_kernel_size: Kernel size to use for the
            convolution op just prior to RPN box predictions.
          first_stage_box_predictor_depth: Output depth for the convolution op
            just prior to RPN box predictions.
          first_stage_minibatch_size: The "batch size" to use for computing the
            objectness and location loss of the region proposal network. This
            "batch size" refers to the number of anchors selected as contributing
            to the loss function for any given image within the image batch and is
            only called "batch_size" due to terminology from the Faster R-CNN paper.
          first_stage_positive_balance_fraction: Fraction of positive examples
            per image for the RPN. The recommended value for Faster RCNN is 0.5.
          first_stage_nms_score_threshold: Score threshold for non max suppression
            for the Region Proposal Network (RPN).  This value is expected to be in
            [0, 1] as it is applied directly after a softmax transformation.  The
            recommended value for Faster R-CNN is 0.
          first_stage_nms_iou_threshold: The Intersection Over Union (IOU) threshold
            for performing Non-Max Suppression (NMS) on the boxes predicted by the
            Region Proposal Network (RPN).
          first_stage_max_proposals: Maximum number of boxes to retain after
            performing Non-Max Suppression (NMS) on the boxes predicted by the
            Region Proposal Network (RPN).
          first_stage_localization_loss_weight: A float
          first_stage_objectness_loss_weight: A float
          initial_crop_size: A single integer indicating the output size
            (width and height are set to be the same) of the initial bilinear
            interpolation based cropping during ROI pooling.
          maxpool_kernel_size: A single integer indicating the kernel size of the
            max pool op on the cropped feature map during ROI pooling.
          maxpool_stride: A single integer indicating the stride of the max pool
            op on the cropped feature map during ROI pooling.
          second_stage_mask_rcnn_box_predictor: Mask R-CNN box predictor to use for
            the second stage.
          second_stage_batch_size: The batch size used for computing the
            classification and refined location loss of the box classifier.  This
            "batch size" refers to the number of proposals selected as contributing
            to the loss function for any given image within the image batch and is
            only called "batch_size" due to terminology from the Faster R-CNN paper.
          second_stage_balance_fraction: Fraction of positive examples to use
            per image for the box classifier. The recommended value for Faster RCNN
            is 0.25.
          second_stage_non_max_suppression_fn: batch_multiclass_non_max_suppression
            callable that takes `boxes`, `scores`, optional `clip_window` and
            optional (kwarg) `mask` inputs (with all other inputs already set)
            and returns a dictionary containing tensors with keys:
            `detection_boxes`, `detection_scores`, `detection_classes`,
            `num_detections`, and (optionally) `detection_masks`. See
            `post_processing.batch_multiclass_non_max_suppression` for the type and
            shape of these tensors.
          second_stage_score_conversion_fn: Callable elementwise nonlinearity
            (that takes tensors as inputs and returns tensors).  This is usually
            used to convert logits to probabilities.
          second_stage_localization_loss_weight: A float indicating the scale factor
            for second stage localization loss.
          second_stage_classification_loss_weight: A float indicating the scale
            factor for second stage classification loss.
          second_stage_classification_loss: Classification loss used by the second
            stage classifier. Either losses.WeightedSigmoidClassificationLoss or
            losses.WeightedSoftmaxClassificationLoss.
          second_stage_mask_prediction_loss_weight: A float indicating the scale
            factor for second stage mask prediction loss. This is applicable only if
            second stage box predictor is configured to predict masks.
          hard_example_miner:  A losses.HardExampleMiner object (can be None).
          parallel_iterations: (Optional) The number of iterations allowed to run
            in parallel for calls to tf.map_fn.
        Raises:
          ValueError: If `second_stage_batch_size` > `first_stage_max_proposals` at
            training time.
          ValueError: If first_stage_anchor_generator is not of type
            grid_anchor_generator.GridAnchorGenerator.
        """
        super(FasterRCNNMetaArch, self).__init__(num_classes=num_classes)

        if is_training and second_stage_batch_size > first_stage_max_proposals:
            raise ValueError('second_stage_batch_size should be no greater than '
                             'first_stage_max_proposals.')
        if not isinstance(first_stage_anchor_generator,
                          grid_anchor_generator.GridAnchorGenerator):
            raise ValueError('first_stage_anchor_generator must be of type '
                             'grid_anchor_generator.GridAnchorGenerator.')

        self._is_training = is_training
        self._is_building = is_building
        self._image_resizer_fn = image_resizer_fn
        self._feature_extractor = feature_extractor
        self._first_stage_only = first_stage_only

        # The first class is reserved as background.
        unmatched_cls_target = tf.constant([1] + self._num_classes * [0], dtype=tf.float32)
        self._proposal_target_assigner = target_assigner.create_target_assigner(
            'FasterRCNN', 'proposal')
        self._detector_target_assigner = target_assigner.create_target_assigner(
            'FasterRCNN', 'detection', unmatched_cls_target=unmatched_cls_target)
        # Both proposal and detector target assigners use the same box coder
        self._box_coder = self._proposal_target_assigner.box_coder

        # (First stage) Region proposal network parameters
        self._first_stage_anchor_generator = first_stage_anchor_generator
        self._first_stage_atrous_rate = first_stage_atrous_rate
        self._first_stage_box_predictor_arg_scope = (
            first_stage_box_predictor_arg_scope)
        self._first_stage_box_predictor_kernel_size = (
            first_stage_box_predictor_kernel_size)
        self._first_stage_box_predictor_depth = first_stage_box_predictor_depth
        self._first_stage_minibatch_size = first_stage_minibatch_size
        self._first_stage_sampler = sampler.BalancedPositiveNegativeSampler(
            positive_fraction=first_stage_positive_balance_fraction)

        self._first_stage_box_predictor = box_predictor.ConvolutionalBoxPredictor(
            self._is_training, num_classes=1,
            conv_hyperparams=self._first_stage_box_predictor_arg_scope,
            min_depth=0, max_depth=0, num_layers_before_predictor=0,
            use_dropout=False, dropout_keep_prob=1.0, kernel_size=1,
            box_code_size=self._box_coder.code_size)

        self._first_stage_nms_score_threshold = first_stage_nms_score_threshold
        self._first_stage_nms_iou_threshold = first_stage_nms_iou_threshold
        self._first_stage_max_proposals = first_stage_max_proposals

        self._first_stage_localization_loss = (
            losses.WeightedSmoothL1LocalizationLoss(anchorwise_output=True))
        self._first_stage_objectness_loss = (
            losses.WeightedSoftmaxClassificationLoss(anchorwise_output=True))
        self._first_stage_loc_loss_weight = first_stage_localization_loss_weight
        self._first_stage_obj_loss_weight = first_stage_objectness_loss_weight

        # Per-region cropping parameters
        self._initial_crop_size = initial_crop_size
        self._maxpool_kernel_size = maxpool_kernel_size
        self._maxpool_stride = maxpool_stride

        self._mask_rcnn_box_predictor = second_stage_mask_rcnn_box_predictor
        self._second_stage_batch_size = second_stage_batch_size  # default=64
        self._second_stage_sampler = sampler.BalancedPositiveNegativeSampler(
            positive_fraction=second_stage_balance_fraction)

        self._second_stage_nms_fn = second_stage_non_max_suppression_fn
        self._second_stage_score_conversion_fn = second_stage_score_conversion_fn

        self._second_stage_localization_loss = (losses.WeightedSmoothL1LocalizationLoss(anchorwise_output=True))
        self._second_stage_classification_loss = second_stage_classification_loss
        self._second_stage_mask_loss = (losses.WeightedSigmoidClassificationLoss(anchorwise_output=True))
        self._second_stage_loc_loss_weight = second_stage_localization_loss_weight
        self._second_stage_cls_loss_weight = second_stage_classification_loss_weight
        self._second_stage_mask_loss_weight = (
            second_stage_mask_prediction_loss_weight)
        self._hard_example_miner = hard_example_miner
        self._parallel_iterations = parallel_iterations

    @property
    def first_stage_feature_extractor_scope(self):
        return 'FirstStageFeatureExtractor'

    @property
    def second_stage_feature_extractor_scope(self):
        return 'SecondStageFeatureExtractor'

    @property
    def first_stage_box_predictor_scope(self):
        return 'FirstStageBoxPredictor'

    @property
    def second_stage_box_predictor_scope(self):
        return 'SecondStageBoxPredictor'

    @property
    def max_num_proposals(self):
        """Max number of proposals (to pad to) for each image in the input batch.

        At training time, this is set to be the `second_stage_batch_size` if hard example miner is not configured,
        else it is set to `first_stage_max_proposals`.
        At inference time, this is always set to `first_stage_max_proposals`.

        Returns:
          A positive integer.
        """
        if self._is_training and not self._hard_example_miner:
            return self._second_stage_batch_size
        return self._first_stage_max_proposals

    def preprocess(self, inputs):
        """ 改变图像尺寸大小，并将像素值取值范围缩放到[-1,1]
        """
        if inputs.dtype is not tf.float32:
            raise ValueError('`preprocess` expects a tf.float32 tensor')
        with tf.name_scope('Preprocessor'):
            # if self._is_building:
            #     resized_inputs = inputs
            #     print(resized_inputs)
            # else:
            #     resized_inputs = tf.map_fn(self._image_resizer_fn,
            #                                elems=inputs,
            #                                dtype=tf.float32,
            #                                parallel_iterations=self._parallel_iterations)
            #     print('heiehie')
            resized_inputs = inputs
            return self._feature_extractor.preprocess(resized_inputs)

    # predict RPN阶段
    def predict_rpn(self, preprocessed_inputs):
        """Predicts unpostprocessed tensors from input tensor.
        """
        # 1.得到 两个feature map、box_list类型的anchor、resize后的image_shape
        (rpn_box_predictor_features, rpn_features_to_crop, anchors_boxlist, image_shape) = \
            self._extract_rpn_feature_maps(preprocessed_inputs)

        # 2.对RPN产生的feature map分别用1*1的卷积核预测 得到anchor的位置偏移量和正负类别
        (rpn_box_encodings, rpn_objectness_predictions_with_background) = \
            self._predict_rpn_proposals(rpn_box_predictor_features)

        # 3.anchor修正与筛选
        clip_window = tf.to_float(tf.stack([0, 0, image_shape[1], image_shape[2]]))
        if self._is_training:
            # 训练时，筛选掉越界的anchor及其对应的box encoding and prediction
            (rpn_box_encodings, rpn_objectness_predictions_with_background, anchors_boxlist) = \
                self._remove_invalid_anchors_and_predictions(
                rpn_box_encodings, rpn_objectness_predictions_with_background,
                anchors_boxlist, clip_window)
        else:
            # 预测时，只筛选掉不在图片size中的anchor，修正越界anchor的大小
            anchors_boxlist = box_list_ops.clip_to_window(anchors_boxlist, clip_window)
        # 4.获取anchor四个坐标=两个角点 绝对坐标值 组成的list类型
        anchors = anchors_boxlist.get()
        # 5.汇总数据
        prediction_dict = {
            'rpn_box_predictor_features': rpn_box_predictor_features,
            'rpn_features_to_crop': rpn_features_to_crop,
            'image_shape': image_shape,
            'rpn_box_encodings': rpn_box_encodings,
            'rpn_objectness_predictions_with_background': rpn_objectness_predictions_with_background,
            'anchors': anchors
        }
        return prediction_dict

    def _extract_rpn_feature_maps(self, preprocessed_inputs):
        """Extracts RPN features.
        """
        # 0.图像大小
        image_shape = tf.shape(preprocessed_inputs)
        # 1.从基础网络提取feature_map
        rpn_features_to_crop = self._feature_extractor.extract_proposal_features(
            preprocessed_inputs, scope=self.first_stage_feature_extractor_scope)
        if not self._is_building:
            feature_map_shape = tf.shape(rpn_features_to_crop)
            # 2.1根据feature_map_shape和anchor_ratio,anchor_size,得到anchor信息结合BoxList类,
            anchors = self._first_stage_anchor_generator.generate(
                [(feature_map_shape[1], feature_map_shape[2])])
            # 2.2 RPN做一次3*3*512卷积后的feature map
            with slim.arg_scope(self._first_stage_box_predictor_arg_scope):
                # 默认值3, 512
                kernel_size = self._first_stage_box_predictor_kernel_size
                rpn_box_predictor_features = slim.conv2d(
                    rpn_features_to_crop,
                    self._first_stage_box_predictor_depth,  # kernel_num
                    kernel_size=[kernel_size, kernel_size],
                    rate=self._first_stage_atrous_rate,
                    activation_fn=tf.nn.relu6)
            return (rpn_box_predictor_features, rpn_features_to_crop,
                    anchors, image_shape)
        else:
            return rpn_features_to_crop

    def _predict_rpn_proposals(self, rpn_box_predictor_features):
        """Adds box predictors to RPN feature map to predict proposals.
        """
        # 合法性检查
        num_anchors_per_location = (
            self._first_stage_anchor_generator.num_anchors_per_location())
        if len(num_anchors_per_location) != 1:
            raise RuntimeError('anchor_generator is expected to generate anchors '
                               'corresponding to a single feature map.')
        # 进行卷积运算 得到预测结果,是一个dict
        box_predictions = self._first_stage_box_predictor.predict(
            rpn_box_predictor_features,
            num_anchors_per_location[0],
            scope=self.first_stage_box_predictor_scope)

        box_encodings = box_predictions[box_predictor.BOX_ENCODINGS]
        objectness_predictions_with_background = box_predictions[box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND]
        # return (tf.squeeze(box_encodings, axis=2), objectness_predictions_with_background)
        return box_encodings, objectness_predictions_with_background

    def _remove_invalid_anchors_and_predictions(self, box_encodings,
                                                objectness_predictions_with_background,
                                                anchors_boxlist,
                                                clip_window):
        """Removes anchors that (partially) fall outside an image.
        """
        pruned_anchors_boxlist, keep_indices = box_list_ops.prune_outside_window(
            anchors_boxlist, clip_window)

        def _batch_gather_kept_indices(predictions_tensor):
            return tf.map_fn(
                partial(tf.gather, indices=keep_indices),
                elems=predictions_tensor,
                dtype=tf.float32,
                parallel_iterations=self._parallel_iterations,
                back_prop=True)

        return (_batch_gather_kept_indices(box_encodings),
                _batch_gather_kept_indices(objectness_predictions_with_background),
                pruned_anchors_boxlist)

    def postprocess_rpn(self, rpn_prediction_dict):
        anchors = rpn_prediction_dict['anchors']
        image_shape = rpn_prediction_dict['image_shape']
        rpn_box_encodings = rpn_prediction_dict['rpn_box_encodings']
        rpn_objectness_predictions_with_background = rpn_prediction_dict['rpn_objectness_predictions_with_background']
        rpn_features_to_crop = rpn_prediction_dict['rpn_features_to_crop']

        with tf.name_scope('FirstStagePostprocessor'):
            # 1.数据预处理
            # 1.1 box_encoding结合anchor得到预测的proposal_boxes
            # 为了满足函数参数要求，增加大小为1的num_class维度,[batch_size, num_anchors, 1, 4]
            rpn_box_encodings = tf.expand_dims(rpn_box_encodings, axis=2)
            # 得到的shape--tensor的维度可以设为None,也可以是固定值。只需要batch的大小
            rpn_encodings_shape = shape_utils.combined_static_and_dynamic_shape(rpn_box_encodings)
            # 将anchor增加一个batch维度并扩展 [1, num_anchors, 4]->[batch_size, num_anchor, 4]
            tiled_anchor_boxes = tf.tile(tf.expand_dims(anchors, 0), [rpn_encodings_shape[0], 1, 1])
            # 根据anchor和box_encode(偏移量)还原真正的box
            proposal_boxes = self._batch_decode_boxes(rpn_box_encodings, tiled_anchor_boxes)
            # 删除大小为1的维度, 删掉num_class维度
            proposal_boxes = tf.squeeze(proposal_boxes, axis=2)

            # 1.2 处理prediction_class, 对axis=-1进行概率归一化--softmax, 去掉背景类别的概率
            # [batch_size, num_anchors]
            rpn_objectness_softmax_without_background = tf.nn.softmax(
                rpn_objectness_predictions_with_background)[:, :, 1]
            # 1.3 窗口--由图片长宽转为左上坐标（0，0）右下坐标（h,w）表示
            clip_window = tf.to_float(tf.stack([0, 0, image_shape[1], image_shape[2]]))

            # 2.非极大值抑制 返回值num_proposal.shape = [batch],一个batch内每张图片的候选框数量
            (proposal_boxes, proposal_scores, _, _, _,
             num_proposals) = post_processing.batch_multiclass_non_max_suppression(
                tf.expand_dims(proposal_boxes, axis=2),
                tf.expand_dims(rpn_objectness_softmax_without_background, axis=2),
                self._first_stage_nms_score_threshold,
                self._first_stage_nms_iou_threshold,
                self._first_stage_max_proposals,
                self._first_stage_max_proposals,
                clip_window=clip_window)
            # 3.训练阶段，进行OHEM 或者 正负推荐框数量降采样
            if self._is_training:
                # 在此节点的梯度停止传播, 搞明白
                proposal_boxes = tf.stop_gradient(proposal_boxes)
                # ohem不需要进行采样
                if not self._hard_example_miner:
                    # 生成label数据
                    # 标注的box四个坐标，resize图像后的绝对坐标，不是归一化的坐标
                    # one_hot编码的类别信息，增加一列负类/背景类
                    (groundtruth_boxlists, groundtruth_classes_with_background_list,
                     _) = self._format_groundtruth_data(image_shape)
                    # 正负样本均衡采样送入第二阶段,这里的num_proposals是min(规定采样数量64，真实样本数量<64)
                    (proposal_boxes, proposal_scores,
                     num_proposals) = self._unpad_proposals_and_sample_box_classifier_batch(
                        proposal_boxes, proposal_scores, num_proposals,
                        groundtruth_boxlists, groundtruth_classes_with_background_list)

            # 4.推荐框坐标归一化
            proposal_boxes_reshaped = tf.reshape(proposal_boxes, [-1, 4])
            normalized_proposal_boxes_reshaped = \
                box_list_ops.to_normalized_coordinates(box_list.BoxList(proposal_boxes_reshaped),
                                                       image_shape[1], image_shape[2], check_range=False).get()
            proposal_boxes = tf.reshape(normalized_proposal_boxes_reshaped, [-1, proposal_boxes.shape[1].value, 4])

            # 5.汇总结果
            detection_dict = {
                'detection_boxes': proposal_boxes,
                'detection_scores': proposal_scores,
                'num_detections': num_proposals,
                'rpn_features_to_crop': rpn_features_to_crop,
                'image_shape': image_shape
            }
        return detection_dict

    def _format_groundtruth_data(self, image_shape):
        """Helper function for preparing groundtruth data for target assignment.
        """
        # 得到图片中标注的坐标,
        groundtruth_boxlists = [
            box_list_ops.to_absolute_coordinates(box_list.BoxList(boxes), image_shape[1], image_shape[2])
            for boxes in self.groundtruth_lists(fields.BoxListFields.boxes)]
        # self.groundtruth_lists()在父类中定义, 得到一个list
        # 在最左边增加一列作为背景类，始终为0
        groundtruth_classes_with_background_list = [
            tf.to_float(tf.pad(one_hot_encoding, [[0, 0], [1, 0]], mode='CONSTANT'))
            for one_hot_encoding in self.groundtruth_lists(fields.BoxListFields.classes)]
        # tf.pad 是扩展/填充的意思，其中[0, 0], [1, 0] 分别代表的是[上，下][左，右]  值为0代表相应边扩展长度为0，
        # 比如上面代码中，左的位置的值为1，代表在左边增加一列，填充是mode=‘CONSTANT‘，代表用0填充，

        groundtruth_masks_list = self._groundtruth_lists.get(
            fields.BoxListFields.masks)
        if groundtruth_masks_list is not None:
            resized_masks_list = []
            for mask in groundtruth_masks_list:
                resized_4d_mask = tf.image.resize_images(
                    tf.expand_dims(mask, axis=3),
                    image_shape[1:3],
                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                    align_corners=True)
                resized_masks_list.append(tf.squeeze(resized_4d_mask, axis=3))
            groundtruth_masks_list = resized_masks_list

        return (groundtruth_boxlists, groundtruth_classes_with_background_list,
                groundtruth_masks_list)

    def _unpad_proposals_and_sample_box_classifier_batch(self, proposal_boxes,
                                                         proposal_scores,
                                                         num_proposals,
                                                         groundtruth_boxlists,
                                                         groundtruth_classes_with_background_list):
        """Unpads proposals and samples a minibatch for second stage.
        """
        single_image_proposal_box_sample = []
        single_image_proposal_score_sample = []
        single_image_num_proposals_sample = []
        # 依次处理单张图片
        for (single_image_proposal_boxes,
             single_image_proposal_scores,
             single_image_num_proposals,
             single_image_groundtruth_boxlist,
             single_image_groundtruth_classes_with_background) in zip(
            tf.unstack(proposal_boxes),
            tf.unstack(proposal_scores),
            tf.unstack(num_proposals),
            groundtruth_boxlists,
            groundtruth_classes_with_background_list):
            # step1 单张图片的推荐框进行数据预处理
            # 1.1
            # tensor.get_shape得到的是静态数据，tf.shape(tensor)得到的是动态数据
            static_shape = single_image_proposal_boxes.get_shape()  # [max_num_proposals, 4]
            sliced_static_shape = tf.TensorShape([tf.Dimension(None), static_shape.dims[-1]])
            # tf.slice() 抽取真实box的部分
            single_image_proposal_boxes = tf.slice(
                single_image_proposal_boxes,
                [0, 0],
                [single_image_num_proposals, -1])
            # shape [None, 4]
            single_image_proposal_boxes.set_shape(sliced_static_shape)
            # 1.2 tf.slice取出前num_proposal个推荐框的score
            single_image_proposal_scores = tf.slice(single_image_proposal_scores,
                                                    [0],
                                                    [single_image_num_proposals])
            # step2 将推荐框bbox score生成一个BoxList类，并将score也加入
            single_image_boxlist = box_list.BoxList(single_image_proposal_boxes)
            single_image_boxlist.add_field(fields.BoxListFields.scores, single_image_proposal_scores)
            # step3 对所有推荐框首先进行正负样本判别，然后抽样，得到抽样之后的BoxList
            sampled_boxlist = self._sample_box_classifier_minibatch(
                single_image_boxlist,
                single_image_groundtruth_boxlist,
                single_image_groundtruth_classes_with_background)
            # step4 采用之后按照阈值进行填充或裁剪
            sampled_padded_boxlist = box_list_ops.pad_or_clip_box_list(
                sampled_boxlist,
                num_boxes=self._second_stage_batch_size)
            # step5 将结果 添加到各自对应的list tf.stack拼接后返回结果
            # 1.num_proposal
            single_image_num_proposals_sample.append(tf.minimum(
                sampled_boxlist.num_boxes(), self._second_stage_batch_size))
            # 2.proposal_box
            bb = sampled_padded_boxlist.get()
            single_image_proposal_box_sample.append(bb)
            # 3.proposal_score
            single_image_proposal_score_sample.append(
                sampled_padded_boxlist.get_field(fields.BoxListFields.scores))

        return (tf.stack(single_image_proposal_box_sample),
                tf.stack(single_image_proposal_score_sample),
                tf.stack(single_image_num_proposals_sample))

    def _sample_box_classifier_minibatch(self, proposal_boxlist, groundtruth_boxlist,
                                         groundtruth_classes_with_background):
        """Samples a mini-batch of proposals to be sent to the box classifier.
        """
        (cls_targets, cls_weights, _, _, _) = self._detector_target_assigner.assign(
            proposal_boxlist, groundtruth_boxlist, groundtruth_classes_with_background)
        # Selects all boxes as candidates if none of them is selected according
        # to cls_weights. This could happen as boxes within certain IOU ranges
        # are ignored. If triggered, the selected boxes will still be ignored
        # during loss computation.
        cls_weights += tf.to_float(tf.equal(tf.reduce_sum(cls_weights), 0))
        positive_indicator = tf.greater(tf.argmax(cls_targets, axis=1), 0)
        # 第二阶段采样器 正样本默认值0.25， 样本不足时 填充0
        # 注意 第一阶段采样器 正负各一半，只在计算rpn阶段loss时使用。
        sampled_indices = self._second_stage_sampler.subsample(
            tf.cast(cls_weights, tf.bool), self._second_stage_batch_size, positive_indicator)

        return box_list_ops.boolean_mask(proposal_boxlist, sampled_indices)

    def predict_second_stage(self, rpn_detection_dict):
        """Predicts the output tensors from second stage of Faster R-CNN.
        """
        proposal_boxes_normalized = rpn_detection_dict['detection_boxes']
        num_proposals = rpn_detection_dict['num_detections']
        rpn_features_to_crop = rpn_detection_dict['rpn_features_to_crop']
        image_shape = rpn_detection_dict['image_shape']
        # 1.根据每个候选框的坐标值,在feature map上裁剪出对应区域进行max_pooling
        flattened_proposal_feature_maps = (self._compute_second_stage_input_feature_maps(
                                            rpn_features_to_crop,
                                            proposal_boxes_normalized))
        # 2.进行卷积运算，提取特征
        box_classifier_features = (self._feature_extractor.extract_box_classifier_features(
                                    flattened_proposal_feature_maps,
                                    scope=self.second_stage_feature_extractor_scope))
        # 3.全连接层预测box位置偏移量和类别 --core->box_predictor
        box_predictions = self._mask_rcnn_box_predictor.predict(
            box_classifier_features,
            num_predictions_per_location=1,
            scope=self.second_stage_box_predictor_scope)
        # 4.预测结果
        # shape[batch*num_proposal, num_class, 4]
        refined_box_encodings = tf.squeeze(box_predictions[box_predictor.BOX_ENCODINGS], axis=1)
        # shape[batch*num_proposal, num_class+1]
        class_predictions_with_background = tf.squeeze(box_predictions[
                                                           box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND], axis=1)
        object_embeddings_with_background = box_predictions['object_embeddings_with_background']
        # 5.候选框映射到原始坐标
        absolute_proposal_boxes = ops.normalized_to_image_coordinates(
            proposal_boxes_normalized, image_shape, self._parallel_iterations)
        # 6.汇总数据
        prediction_dict = {
            'box_classifier_features': box_classifier_features,
            'refined_box_encodings': refined_box_encodings,
            'class_predictions_with_background': class_predictions_with_background,
            'object_embeddings_with_background': object_embeddings_with_background,
            'num_proposals': num_proposals,
            'proposal_boxes': absolute_proposal_boxes,
            'proposal_boxes_normalized': proposal_boxes_normalized,
            'image_shape': image_shape
        }

        return prediction_dict

    def predict_second_stage_building(self, rpn_features_to_crop):
        # 2.根据每个候选框的坐标值,在feature map上裁剪出对应区域进行max_pooling
        # 裁剪区域首先resize(14,14), max_pooling(kernel=(2,2),stride=2)
        # 最终产生 batch*num_proposal 个(7,7,depth)feature_map
        flattened_proposal_feature_maps = self._compute_second_stage_input_feature_maps(rpn_features_to_crop, None)
        # 3.进行卷积运算，提取特征
        box_classifier_features = (self._feature_extractor.extract_box_classifier_features(
            flattened_proposal_feature_maps,
            scope=self.second_stage_feature_extractor_scope))

        # 4.全连接层运算 + l2 norm
        box_embeddings = self._mask_rcnn_box_predictor.predict(
            box_classifier_features,
            num_predictions_per_location=1,
            scope=self.second_stage_box_predictor_scope)
        return box_embeddings

    def _compute_second_stage_input_feature_maps(self, features_to_crop, proposal_boxes_normalized):
        """Crops to a set of proposals from the feature map for a batch of images.
        """

        # 每个候选框对应batch_size的索引
        def get_box_inds(proposals):
            proposals_shape = proposals.get_shape().as_list()

            if any(dim is None for dim in proposals_shape):
                proposals_shape = tf.shape(proposals)
            # [batch_size, num_proposals]
            ones_mat = tf.ones(proposals_shape[:2], dtype=tf.int32)
            # tf.range(start,limit,delta)产生一个序列，shape为[limit]
            # tf.expand_dim(tensor,axis),给tensor扩展一个维度
            # 故multiplier的shape=[batch_size,1], value=[[0],[1]...[batch_size-1]]
            multiplier = tf.expand_dims(tf.range(start=0, limit=proposals_shape[0]), 1)
            # shape = [batch_size, num_proposals] -> 1-D
            # value[0,0,..0,1,1..1,....batch_size,batch_size..batch_size]
            return tf.reshape(ones_mat * multiplier, [-1])  # shape=[-1]时， flattens into 1-D.

        # 裁剪出k个子区域并resize到目标形状(14,14),最终shape(batch*num_proposal,14,14,depth)
        if self._is_building:
            cropped_regions = tf.image.resize_bilinear(features_to_crop,
                                                       (self._initial_crop_size, self._initial_crop_size))
        else:
            cropped_regions = tf.image.crop_and_resize(
                features_to_crop,
                self._flatten_first_two_dimensions(proposal_boxes_normalized),  # 将前两维[batch, num_anchor]展开
                get_box_inds(proposal_boxes_normalized),  # 确定候选框所在的batch, 找到对应的feature_map to crop
                (self._initial_crop_size, self._initial_crop_size))
        # kernel size(2,2), stride=2 最大值pool
        # 得到(7,7)的特征区域
        # return slim.max_pool2d(
        #     cropped_regions,
        #     [self._maxpool_kernel_size, self._maxpool_kernel_size],
        #     stride=self._maxpool_stride)
        return cropped_regions

    def _flatten_first_two_dimensions(self, inputs):
        """Flattens `K-d` tensor along batch dimension to be a `(K-1)-d` tensor.
        """
        combined_shape = shape_utils.combined_static_and_dynamic_shape(inputs)
        flattened_shape = tf.stack([combined_shape[0] * combined_shape[1]] +
                                   combined_shape[2:])
        return tf.reshape(inputs, flattened_shape)

    # postprocess阶段 对predict结果进行筛选
    def postprocess_box_classifier(self, final_prediction_dict):
        """Converts predictions from the second stage box classifier to detections.
        """
        refined_box_encodings = final_prediction_dict['refined_box_encodings']
        class_predictions_with_background = final_prediction_dict['class_predictions_with_background']
        proposal_boxes = final_prediction_dict['proposal_boxes']  # ab_proposal_boxes
        num_proposals = final_prediction_dict['num_proposals']
        image_shape = final_prediction_dict['image_shape']

        with tf.name_scope('SecondStagePostprocessor'):
            # 1.将batch_size*num_proposal维度拆分为 batch_size,num_proposals
            refined_box_encodings_batch = tf.reshape(refined_box_encodings,
                                                     [-1, self.max_num_proposals,
                                                      self.num_classes,
                                                      self._box_coder.code_size])
            class_predictions_with_background_batch = tf.reshape(
                class_predictions_with_background,
                [-1, self.max_num_proposals, self.num_classes + 1]
            )
            # 2.将预测的目标框偏移量，转换成图像上的坐标
            refined_decoded_boxes_batch = self._batch_decode_boxes(refined_box_encodings_batch, proposal_boxes)
            # 3.将预测类别的值转换成softmax
            class_predictions_with_background_batch = (
                self._second_stage_score_conversion_fn(class_predictions_with_background_batch))
            # 去掉负类
            class_predictions_batch = tf.reshape(
                tf.slice(class_predictions_with_background_batch, [0, 0, 1], [-1, -1, -1]),
                [-1, self.max_num_proposals, self.num_classes])

            clip_window = tf.to_float(tf.stack([0, 0, image_shape[1], image_shape[2]]))

            # 5.NMS
            (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks, _,
             num_detections) = self._second_stage_nms_fn(
                refined_decoded_boxes_batch,
                class_predictions_batch,
                clip_window=clip_window,
                change_coordinate_frame=True,
                num_valid_boxes=num_proposals,
                masks=None)

            detections = {'detection_boxes': nmsed_boxes,
                          'detection_scores': nmsed_scores,
                          'detection_classes': nmsed_classes,
                          'num_detections': tf.to_float(num_detections)}

        return detections

    def _batch_decode_boxes(self, box_encodings, anchor_boxes):
        """Decodes box encodings with respect to the anchor boxes.
        """
        combined_shape = shape_utils.combined_static_and_dynamic_shape(box_encodings)
        num_classes = combined_shape[2]
        # anchor_box增加class一个维度，保持相同的形状
        tiled_anchor_boxes = tf.tile(tf.expand_dims(anchor_boxes, 2), [1, 1, num_classes, 1])
        # reshape（数量，位置）
        tiled_anchors_boxlist = box_list.BoxList(tf.reshape(tiled_anchor_boxes, [-1, 4]))
        # 同样reshape
        decoded_boxes = self._box_coder.decode(
            tf.reshape(box_encodings, [-1, self._box_coder.code_size]), tiled_anchors_boxlist)

        return tf.reshape(decoded_boxes.get(), tf.stack([combined_shape[0], combined_shape[1], num_classes, 4]))

    # loss计算
    def loss(self, prediction_dict, scope=None):
        """Compute scalar loss tensors given prediction tensors.
        """
        with tf.name_scope(scope, 'Loss', prediction_dict.values()):
            # 第一阶段loss
            loss_dict = self.loss_rpn(prediction_dict)
            # 第二阶段loss
            if not self._first_stage_only:
                loss_dict.update(self.loss_box_classifier(prediction_dict))
        return loss_dict

    def loss_rpn(self, rpn_prediction_dict):
        """Computes scalar RPN loss tensors.
        """
        rpn_box_encodings = rpn_prediction_dict['rpn_box_encodings']
        rpn_objectness_predictions_with_background = rpn_prediction_dict['rpn_objectness_predictions_with_background']
        anchors = rpn_prediction_dict['anchors']
        groundtruth_boxlists, _, _ = self._format_groundtruth_data(rpn_prediction_dict['image_shape'])

        with tf.name_scope('RPNLoss'):
            # step1 根据标注为每个anchor确定所属类别和偏移量，即整个模型的学习目标
            # 传入BoxList包含一个feature map产生的全部anchor 不是List(BoxList(anchor))
            # 因为所有的标注信息均为正样本
            # 所以不采用_format_groundtruth_data得到的groundtruth_classes_with_background_list
            (batch_cls_targets, batch_cls_weights, batch_reg_targets, batch_reg_weights, _) = \
                target_assigner.batch_assign_targets(
                    self._proposal_target_assigner,
                    box_list.BoxList(anchors),
                    groundtruth_boxlists,
                    len(groundtruth_boxlists) * [None])
            # batch_cls_targets shape=[batch_size, num_anchor, num_class] 如果只有一个类，则去掉那一列
            batch_cls_targets = tf.squeeze(batch_cls_targets, axis=2)

            # step2 部分采样计算loss batch_size默认值是256，正负样本各一半
            def _minibatch_subsample_fn(inputs):
                cls_targets, cls_weights = inputs
                return self._first_stage_sampler.subsample(
                    tf.cast(cls_weights, tf.bool),
                    self._first_stage_minibatch_size,
                    tf.cast(cls_targets, tf.bool))
            # 得到采样的位置索引值
            batch_sampled_indices = tf.to_float(tf.map_fn(
                _minibatch_subsample_fn,
                [batch_cls_targets, batch_cls_weights],
                dtype=tf.bool,
                parallel_iterations=self._parallel_iterations,
                back_prop=True))

            # Normalize by number of examples in sampled minibatch
            # 得到每张图片的采样数量，用于计算均值
            normalizer = tf.reduce_sum(batch_sampled_indices, axis=1)
            # tf.one_hot
            # shape[batch,num_anchor,2]
            batch_one_hot_targets = tf.one_hot(tf.to_int32(batch_cls_targets), depth=2)
            # 采样对应的回归权重 tf.multiply矩阵中对应位置相乘，则最终保留采样位置的权重值
            sampled_reg_indices = tf.multiply(batch_sampled_indices, batch_reg_weights)
            # step2.1 每个采样数据的loss
            localization_losses = self._first_stage_localization_loss(
                rpn_box_encodings, batch_reg_targets, weights=sampled_reg_indices)
            # 这里的没有用batch_cls_weight分类权重，直接使用索引结果batch_sampled_indices，因为默认都为1
            objectness_losses = self._first_stage_objectness_loss(
                rpn_objectness_predictions_with_background,
                batch_one_hot_targets, weights=batch_sampled_indices)
            # step2.2 整个batch的平均loss
            # 首先对第2维求和，得到一张图片内所有预测框的loss总和，除以采样数量256归一化
            # 两个loss的归一化 与论文中稍有不同
            localization_loss = tf.reduce_mean(tf.reduce_sum(localization_losses, axis=1) / normalizer)
            objectness_loss = tf.reduce_mean(tf.reduce_sum(objectness_losses, axis=1) / normalizer)
            # step2.3 乘上权重
            loss_dict = {}
            with tf.name_scope('localization_loss'):
                loss_dict['first_stage_localization_loss'] = (
                        self._first_stage_loc_loss_weight * localization_loss)
            with tf.name_scope('objectness_loss'):
                loss_dict['first_stage_objectness_loss'] = (
                        self._first_stage_obj_loss_weight * objectness_loss)
        return loss_dict

    def loss_box_classifier(self, final_prediction_dict):
        """Computes scalar box classifier loss tensors.
        """

        object_embeddings_with_background = final_prediction_dict['object_embeddings_with_background']
        refined_box_encodings = final_prediction_dict['refined_box_encodings']
        class_predictions_with_background = final_prediction_dict['class_predictions_with_background']
        proposal_boxes = final_prediction_dict['proposal_boxes']
        num_proposals = final_prediction_dict['num_proposals']

        groundtruth_boxlists, groundtruth_classes_with_background_list, _ \
            = self._format_groundtruth_data(final_prediction_dict['image_shape'])

        # cls_targets_with_background = self._flatten_first_two_dimensions(batch_cls_targets_with_background)  # label
        # cls_targets_with_background = tf.reduce_max(cls_targets_with_background, axis=1)

        with tf.name_scope('BoxClassifierLoss'):
            # 得到一个索引矩阵[batch,64], proposals时值为1，num_proposals为真实推荐数量，
            paddings_indicator = self._padded_batched_proposals_indicator(num_proposals, self.max_num_proposals)
            # 将每张图片内的proposal_boxes组成BoxList,所有图片的BoxList组成list
            # proposal_boxes未满足数量时，填充0，
            proposal_boxlists = [
                box_list.BoxList(proposal_boxes_single_image)
                for proposal_boxes_single_image in tf.unstack(proposal_boxes)]
            batch_size = len(proposal_boxlists)
            # 将没有proposal_box的图片改为1，防止除数为0
            # num_proposals_or_one的shape为[batch,1]
            num_proposals_or_one = tf.to_float(
                tf.expand_dims(tf.maximum(num_proposals, tf.ones_like(num_proposals)), 1))
            # 首先将num_proposals_or_one的shape改为[batch,64] 将value乘以batch_size
            # 因为后面的运算是先对每个proposal的loss进行除法再进行求和来得到平均loss
            normalizer = tf.tile(num_proposals_or_one, [1, self.max_num_proposals]) * batch_size
            # 设置IoU阈值为0.5，对RPN的proposal进行分类，大于的为正类，小于的为负类
            (batch_cls_targets_with_background, batch_cls_weights, batch_reg_targets,
             batch_reg_weights, _) = target_assigner.batch_assign_targets(
                self._detector_target_assigner, proposal_boxlists,
                groundtruth_boxlists, groundtruth_classes_with_background_list)

            # We only predict refined location encodings for the non background
            # classes, but we now pad it to make it compatible with the class
            # predictions
            # 保持最后一维类别不变，展开所有的proposal_box
            flat_cls_targets_with_background = tf.reshape(batch_cls_targets_with_background,
                                                          [batch_size * self.max_num_proposals, -1])
            refined_box_encodings_with_background = tf.pad(refined_box_encodings, [[0, 0], [1, 0], [0, 0]])
            # For anchors with multiple labels, picks refined_location_encodings
            # for just one class to avoid over-counting for regression loss and
            # (optionally) mask loss.
            # 多标签值选择一个计算回归,损失避免重复
            one_hot_flat_cls_targets_with_background = tf.argmax(flat_cls_targets_with_background, axis=1)
            one_hot_flat_cls_targets_with_background = tf.one_hot(
                one_hot_flat_cls_targets_with_background,
                flat_cls_targets_with_background.get_shape()[1])
            refined_box_encodings_masked_by_class_targets = tf.boolean_mask(
                refined_box_encodings_with_background,
                tf.greater(one_hot_flat_cls_targets_with_background, 0))
            # reshape
            # 第二阶段的预测结果 以所有的推荐框作为new_batch, 在此处重新按照图片数量组成batch
            class_predictions_with_background = tf.reshape(
                class_predictions_with_background,
                [batch_size, self.max_num_proposals, -1])
            reshaped_refined_box_encodings = tf.reshape(
                refined_box_encodings_masked_by_class_targets,
                [batch_size, -1, 4])
            # 计算loss
            # normalizer的shape[batch,num_proposals],value[num_proposals*batch_size]
            # 此步骤提前进行除法运算，在后面进行求和，最终得到batch平均
            second_stage_loc_losses = self._second_stage_localization_loss(
                reshaped_refined_box_encodings,
                batch_reg_targets, weights=batch_reg_weights) / normalizer
            # 没有指定的情况下 默认使用softmax, 参数为true
            # batch_cls_targets_with_background one-hot编码
            # shape[batch,num_proposal,num_class+1]
            # class_weight shape[num_class+1]
            label_weight = tf.ones(shape=[self._num_classes + 1]) * 2
            batch_cls_targets_with_background = batch_cls_targets_with_background * label_weight

            second_stage_cls_losses = self._second_stage_classification_loss(
                class_predictions_with_background,
                batch_cls_targets_with_background,
                weights=batch_cls_weights) / normalizer
            # tf.boolean_mask和tf.gather类似，取为true位置的值
            # 求得batch内的平均
            second_stage_loc_loss = tf.reduce_sum(tf.boolean_mask(second_stage_loc_losses, paddings_indicator))
            second_stage_cls_loss = tf.reduce_sum(tf.boolean_mask(second_stage_cls_losses, paddings_indicator))

            if self._hard_example_miner:
                (second_stage_loc_loss, second_stage_cls_loss
                 ) = self._unpad_proposals_and_apply_hard_mining(
                    proposal_boxlists, second_stage_loc_losses,
                    second_stage_cls_losses, num_proposals)
            loss_dict = {}
            with tf.name_scope('localization_loss'):
                loss_dict['second_stage_localization_loss'] = (
                        self._second_stage_loc_loss_weight * second_stage_loc_loss)

            with tf.name_scope('classification_loss'):
                loss_dict['second_stage_classification_loss'] = (
                        self._second_stage_cls_loss_weight * second_stage_cls_loss)
            # TODO: triplet loss
            fla_cls_targets_with_background = tf.reshape(batch_cls_targets_with_background,
                                                         [batch_size * self.max_num_proposals, -1])
            fla_cls_targets_with_background = tf.argmax(fla_cls_targets_with_background, axis=1)
            second_stage_metric_loss, fraction = self._batch_all_triplet_loss(
                fla_cls_targets_with_background, object_embeddings_with_background, margin=0.3)
            with tf.name_scope('metric_loss'):
                loss_dict['second_stage_metric_loss'] = second_stage_metric_loss
                loss_dict['triplet_label'] = fla_cls_targets_with_background
        return loss_dict

    def _pairwise_distances(self, embeddings, squared=False):
        """Compute the 2D matrix of distances between all the embeddings.

        Args:
            embeddings: tensor of shape (batch_size, embed_dim)
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.

        Returns:
            pairwise_distances: tensor of shape (batch_size, batch_size)
        """
        # Get the dot product between all embeddings
        # shape (batch_size, batch_size)
        dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

        # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
        # This also provides more numerical stability (the diagonal of the result will be exactly 0).
        # shape (batch_size,)
        # 取对角线元素，是每个向量L2 norm的平方 组成一维数组
        square_norm = tf.diag_part(dot_product)

        # Compute the pairwise distance matrix as we have:
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances = tf.maximum(distances, 0.0)

        if not squared:
            # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
            # we need to add a small epsilon where distances == 0.0
            mask = tf.to_float(tf.equal(distances, 0.0))
            distances = distances + mask * 1e-16

            distances = tf.sqrt(distances)

            # Correct the epsilon added: set the distances on the mask to be exactly 0.0
            distances = distances * (1.0 - mask)

        return distances

    def _get_anchor_positive_triplet_mask(self, labels):
        """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]

        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Check that i and j are distinct
        indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
        indices_not_equal = tf.logical_not(indices_equal)

        # Check if labels[i] == labels[j]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

        # Combine the two masks
        mask = tf.logical_and(indices_not_equal, labels_equal)

        return mask

    def _get_anchor_negative_triplet_mask(self, labels):
        """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]

        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Check if labels[i] != labels[k]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

        mask = tf.logical_not(labels_equal)

        return mask

    def _get_triplet_mask(self, labels):
        """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

        A triplet (i, j, k) is valid if:
            - i, j, k are distinct
            - labels[i] == labels[j] and labels[i] != labels[k]

        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        """
        # Check that i, j and k are distinct
        indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
        indices_not_equal = tf.logical_not(indices_equal)
        i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
        i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
        j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

        distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

        # Check if labels[i] == labels[j] and labels[i] != labels[k]
        label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        i_equal_j = tf.expand_dims(label_equal, 2)
        i_equal_k = tf.expand_dims(label_equal, 1)

        valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

        # Combine the two masks
        mask = tf.logical_and(distinct_indices, valid_labels)

        return mask

    def _batch_all_triplet_loss(self, labels, embeddings, margin, squared=False):
        """Build the triplet loss over a batch of embeddings.

        We generate all the valid triplets and average the loss over the positive ones.

        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.

        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        pairwise_dist = self._pairwise_distances(embeddings, squared=squared)

        # shape (batch_size, batch_size, 1)
        anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
        assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
        # shape (batch_size, 1, batch_size)
        anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
        assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        mask = self._get_triplet_mask(labels)
        mask = tf.to_float(mask)
        triplet_loss = tf.multiply(mask, triplet_loss)

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss = tf.maximum(triplet_loss, 0.0)

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
        num_positive_triplets = tf.reduce_sum(valid_triplets)
        num_valid_triplets = tf.reduce_sum(mask)
        fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

        return triplet_loss, fraction_positive_triplets

    def _batch_hard_triplet_loss(self, labels, embeddings, margin, squared=False):
        """Build the triplet loss over a batch of embeddings.

        For each anchor, we get the hardest positive and hardest negative to form a triplet.

        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.

        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        pairwise_dist = self._pairwise_distances(embeddings, squared=squared)

        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = self._get_anchor_positive_triplet_mask(labels)
        mask_anchor_positive = tf.to_float(mask_anchor_positive)

        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

        # shape (batch_size, 1)
        hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
        tf.summary.scalar("hardest_positive_dist", tf.reduce_mean(hardest_positive_dist))

        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        mask_anchor_negative = self._get_anchor_negative_triplet_mask(labels)
        mask_anchor_negative = tf.to_float(mask_anchor_negative)

        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

        # shape (batch_size,)
        hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
        tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)

        # Get final mean triplet loss
        triplet_loss = tf.reduce_mean(triplet_loss)

        return triplet_loss

    def _padded_batched_proposals_indicator(self,
                                            num_proposals,
                                            max_num_proposals):
        """Creates indicator matrix of non-pad elements of padded batch proposals.

        Args:
          num_proposals: Tensor of type tf.int32 with shape [batch_size].
          max_num_proposals: Maximum number of proposals per image (integer).

        Returns:
          A Tensor of type tf.bool with shape [batch_size, max_num_proposals].
        """
        batch_size = tf.size(num_proposals)
        tiled_num_proposals = tf.tile(tf.expand_dims(num_proposals, 1), [1, max_num_proposals])
        tiled_proposal_index = tf.tile(tf.expand_dims(tf.range(max_num_proposals), 0), [batch_size, 1])
        return tf.greater(tiled_num_proposals, tiled_proposal_index)

    def _unpad_proposals_and_apply_hard_mining(self,
                                               proposal_boxlists,
                                               second_stage_loc_losses,
                                               second_stage_cls_losses,
                                               num_proposals):
        """Unpads proposals and applies hard mining.

        Args:
          proposal_boxlists: A list of `batch_size` BoxLists each representing
            `self.max_num_proposals` representing decoded proposal bounding boxes
            for each image.
          second_stage_loc_losses: A Tensor of type `float32`. A tensor of shape
            `[batch_size, self.max_num_proposals]` representing per-anchor
            second stage localization loss values.
          second_stage_cls_losses: A Tensor of type `float32`. A tensor of shape
            `[batch_size, self.max_num_proposals]` representing per-anchor
            second stage classification loss values.
          num_proposals: A Tensor of type `int32`. A 1-D tensor of shape [batch]
            representing the number of proposals predicted for each image in
            the batch.

        Returns:
          second_stage_loc_loss: A scalar float32 tensor representing the second
            stage localization loss.
          second_stage_cls_loss: A scalar float32 tensor representing the second
            stage classification loss.
        """
        for (proposal_boxlist, single_image_loc_loss, single_image_cls_loss,
             single_image_num_proposals) in zip(
            proposal_boxlists,
            tf.unstack(second_stage_loc_losses),
            tf.unstack(second_stage_cls_losses),
            tf.unstack(num_proposals)):
            proposal_boxlist = box_list.BoxList(
                tf.slice(proposal_boxlist.get(),
                         [0, 0], [single_image_num_proposals, -1]))
            single_image_loc_loss = tf.slice(single_image_loc_loss,
                                             [0], [single_image_num_proposals])
            single_image_cls_loss = tf.slice(single_image_cls_loss,
                                             [0], [single_image_num_proposals])
            return self._hard_example_miner(
                location_losses=tf.expand_dims(single_image_loc_loss, 0),
                cls_losses=tf.expand_dims(single_image_cls_loss, 0),
                decoded_boxlist_list=[proposal_boxlist])

    def restore_map(self, from_detection_checkpoint=True):
        """Returns a map of variables to load from a foreign checkpoint.

        See parent class for details.

        Args:
          from_detection_checkpoint: whether to restore from a full detection
            checkpoint (with compatible variable names) or to restore from a
            classification checkpoint for initialization prior to training.

        Returns:
          A dict mapping variable names (to load from a checkpoint) to variables in
          the model graph.
        """
        # 完整模型的ckpt文件，还是基础网络的ckpt
        if not from_detection_checkpoint:
            return self._feature_extractor.restore_from_classification_checkpoint_fn(
                self.first_stage_feature_extractor_scope,
                self.second_stage_feature_extractor_scope)

        variables_to_restore = tf.global_variables()
        variables_to_restore.append(tf.train.get_or_create_global_step())
        # Only load feature extractor variables to be consistent with loading from
        # a classification checkpoint.

        feature_extractor_variables = tf.contrib.framework.filter_variables(
            variables_to_restore,
            include_patterns=[self.first_stage_feature_extractor_scope, self.second_stage_feature_extractor_scope])
        var_map = {var.op.name: var for var in feature_extractor_variables}

        # ignore_name = 'SecondStageBoxPredictor/BoxEmbeddings/biases'
        # print(var_map.get(ignore_name, 'no name'))
        # del var_map[ignore_name]
        # print(var_map.get(ignore_name,'no name'))
        return var_map
