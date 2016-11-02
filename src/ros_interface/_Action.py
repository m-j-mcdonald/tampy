# This Python file uses the following encoding: utf-8
"""autogenerated by genpy from baxter_core_msgs/Action.msg. Do not edit."""
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct

import baxter_core_msgs.msg

class Action(genpy.Message):
  _md5sum = "dbd6563abeb93158bae1ca356702585e"
  _type = "baxter_core_msgs/Action"
  _has_header = False #flag to mark the presence of a Header object
  _full_text = """int32 step_num
string name
int32[] active_timesteps
Parameter[] parameters
Predicate[] predicates

================================================================================
MSG: baxter_core_msgs/Parameter
string type
bool is_symbol
FloatArray[] lArmPose
FloatArray[] rArmPose
FloatArray[] lGripper
FloatArray[] rGripper
FloatArray[] pose
FloatArray[] value
FloatArray[] rotation

================================================================================
MSG: baxter_core_msgs/FloatArray
float64[] data
================================================================================
MSG: baxter_core_msgs/Predicate
string type
string name
Parameter[] params
string[] param_types"""
  __slots__ = ['step_num','name','active_timesteps','parameters','predicates']
  _slot_types = ['int32','string','int32[]','baxter_core_msgs/Parameter[]','baxter_core_msgs/Predicate[]']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       step_num,name,active_timesteps,parameters,predicates

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(Action, self).__init__(*args, **kwds)
      #message fields cannot be None, assign default values for those that are
      if self.step_num is None:
        self.step_num = 0
      if self.name is None:
        self.name = ''
      if self.active_timesteps is None:
        self.active_timesteps = []
      if self.parameters is None:
        self.parameters = []
      if self.predicates is None:
        self.predicates = []
    else:
      self.step_num = 0
      self.name = ''
      self.active_timesteps = []
      self.parameters = []
      self.predicates = []

  def _get_types(self):
    """
    internal API method
    """
    return self._slot_types

  def serialize(self, buff):
    """
    serialize message into buffer
    :param buff: buffer, ``StringIO``
    """
    try:
      buff.write(_struct_i.pack(self.step_num))
      _x = self.name
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      if python3:
        buff.write(struct.pack('<I%sB'%length, length, *_x))
      else:
        buff.write(struct.pack('<I%ss'%length, length, _x))
      length = len(self.active_timesteps)
      buff.write(_struct_I.pack(length))
      pattern = '<%si'%length
      buff.write(struct.pack(pattern, *self.active_timesteps))
      length = len(self.parameters)
      buff.write(_struct_I.pack(length))
      for val1 in self.parameters:
        _x = val1.type
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        if python3:
          buff.write(struct.pack('<I%sB'%length, length, *_x))
        else:
          buff.write(struct.pack('<I%ss'%length, length, _x))
        buff.write(_struct_B.pack(val1.is_symbol))
        length = len(val1.lArmPose)
        buff.write(_struct_I.pack(length))
        for val2 in val1.lArmPose:
          length = len(val2.data)
          buff.write(_struct_I.pack(length))
          pattern = '<%sd'%length
          buff.write(struct.pack(pattern, *val2.data))
        length = len(val1.rArmPose)
        buff.write(_struct_I.pack(length))
        for val2 in val1.rArmPose:
          length = len(val2.data)
          buff.write(_struct_I.pack(length))
          pattern = '<%sd'%length
          buff.write(struct.pack(pattern, *val2.data))
        length = len(val1.lGripper)
        buff.write(_struct_I.pack(length))
        for val2 in val1.lGripper:
          length = len(val2.data)
          buff.write(_struct_I.pack(length))
          pattern = '<%sd'%length
          buff.write(struct.pack(pattern, *val2.data))
        length = len(val1.rGripper)
        buff.write(_struct_I.pack(length))
        for val2 in val1.rGripper:
          length = len(val2.data)
          buff.write(_struct_I.pack(length))
          pattern = '<%sd'%length
          buff.write(struct.pack(pattern, *val2.data))
        length = len(val1.pose)
        buff.write(_struct_I.pack(length))
        for val2 in val1.pose:
          length = len(val2.data)
          buff.write(_struct_I.pack(length))
          pattern = '<%sd'%length
          buff.write(struct.pack(pattern, *val2.data))
        length = len(val1.value)
        buff.write(_struct_I.pack(length))
        for val2 in val1.value:
          length = len(val2.data)
          buff.write(_struct_I.pack(length))
          pattern = '<%sd'%length
          buff.write(struct.pack(pattern, *val2.data))
        length = len(val1.rotation)
        buff.write(_struct_I.pack(length))
        for val2 in val1.rotation:
          length = len(val2.data)
          buff.write(_struct_I.pack(length))
          pattern = '<%sd'%length
          buff.write(struct.pack(pattern, *val2.data))
      length = len(self.predicates)
      buff.write(_struct_I.pack(length))
      for val1 in self.predicates:
        _x = val1.type
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        if python3:
          buff.write(struct.pack('<I%sB'%length, length, *_x))
        else:
          buff.write(struct.pack('<I%ss'%length, length, _x))
        _x = val1.name
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        if python3:
          buff.write(struct.pack('<I%sB'%length, length, *_x))
        else:
          buff.write(struct.pack('<I%ss'%length, length, _x))
        length = len(val1.params)
        buff.write(_struct_I.pack(length))
        for val2 in val1.params:
          _x = val2.type
          length = len(_x)
          if python3 or type(_x) == unicode:
            _x = _x.encode('utf-8')
            length = len(_x)
          if python3:
            buff.write(struct.pack('<I%sB'%length, length, *_x))
          else:
            buff.write(struct.pack('<I%ss'%length, length, _x))
          buff.write(_struct_B.pack(val2.is_symbol))
          length = len(val2.lArmPose)
          buff.write(_struct_I.pack(length))
          for val3 in val2.lArmPose:
            length = len(val3.data)
            buff.write(_struct_I.pack(length))
            pattern = '<%sd'%length
            buff.write(struct.pack(pattern, *val3.data))
          length = len(val2.rArmPose)
          buff.write(_struct_I.pack(length))
          for val3 in val2.rArmPose:
            length = len(val3.data)
            buff.write(_struct_I.pack(length))
            pattern = '<%sd'%length
            buff.write(struct.pack(pattern, *val3.data))
          length = len(val2.lGripper)
          buff.write(_struct_I.pack(length))
          for val3 in val2.lGripper:
            length = len(val3.data)
            buff.write(_struct_I.pack(length))
            pattern = '<%sd'%length
            buff.write(struct.pack(pattern, *val3.data))
          length = len(val2.rGripper)
          buff.write(_struct_I.pack(length))
          for val3 in val2.rGripper:
            length = len(val3.data)
            buff.write(_struct_I.pack(length))
            pattern = '<%sd'%length
            buff.write(struct.pack(pattern, *val3.data))
          length = len(val2.pose)
          buff.write(_struct_I.pack(length))
          for val3 in val2.pose:
            length = len(val3.data)
            buff.write(_struct_I.pack(length))
            pattern = '<%sd'%length
            buff.write(struct.pack(pattern, *val3.data))
          length = len(val2.value)
          buff.write(_struct_I.pack(length))
          for val3 in val2.value:
            length = len(val3.data)
            buff.write(_struct_I.pack(length))
            pattern = '<%sd'%length
            buff.write(struct.pack(pattern, *val3.data))
          length = len(val2.rotation)
          buff.write(_struct_I.pack(length))
          for val3 in val2.rotation:
            length = len(val3.data)
            buff.write(_struct_I.pack(length))
            pattern = '<%sd'%length
            buff.write(struct.pack(pattern, *val3.data))
        length = len(val1.param_types)
        buff.write(_struct_I.pack(length))
        for val2 in val1.param_types:
          length = len(val2)
          if python3 or type(val2) == unicode:
            val2 = val2.encode('utf-8')
            length = len(val2)
          if python3:
            buff.write(struct.pack('<I%sB'%length, length, *val2))
          else:
            buff.write(struct.pack('<I%ss'%length, length, val2))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize(self, str):
    """
    unpack serialized message in str into this message instance
    :param str: byte array of serialized message, ``str``
    """
    try:
      if self.parameters is None:
        self.parameters = None
      if self.predicates is None:
        self.predicates = None
      end = 0
      start = end
      end += 4
      (self.step_num,) = _struct_i.unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.name = str[start:end].decode('utf-8')
      else:
        self.name = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%si'%length
      start = end
      end += struct.calcsize(pattern)
      self.active_timesteps = struct.unpack(pattern, str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.parameters = []
      for i in range(0, length):
        val1 = baxter_core_msgs.msg.Parameter()
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          val1.type = str[start:end].decode('utf-8')
        else:
          val1.type = str[start:end]
        start = end
        end += 1
        (val1.is_symbol,) = _struct_B.unpack(str[start:end])
        val1.is_symbol = bool(val1.is_symbol)
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        val1.lArmPose = []
        for i in range(0, length):
          val2 = baxter_core_msgs.msg.FloatArray()
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          pattern = '<%sd'%length
          start = end
          end += struct.calcsize(pattern)
          val2.data = struct.unpack(pattern, str[start:end])
          val1.lArmPose.append(val2)
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        val1.rArmPose = []
        for i in range(0, length):
          val2 = baxter_core_msgs.msg.FloatArray()
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          pattern = '<%sd'%length
          start = end
          end += struct.calcsize(pattern)
          val2.data = struct.unpack(pattern, str[start:end])
          val1.rArmPose.append(val2)
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        val1.lGripper = []
        for i in range(0, length):
          val2 = baxter_core_msgs.msg.FloatArray()
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          pattern = '<%sd'%length
          start = end
          end += struct.calcsize(pattern)
          val2.data = struct.unpack(pattern, str[start:end])
          val1.lGripper.append(val2)
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        val1.rGripper = []
        for i in range(0, length):
          val2 = baxter_core_msgs.msg.FloatArray()
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          pattern = '<%sd'%length
          start = end
          end += struct.calcsize(pattern)
          val2.data = struct.unpack(pattern, str[start:end])
          val1.rGripper.append(val2)
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        val1.pose = []
        for i in range(0, length):
          val2 = baxter_core_msgs.msg.FloatArray()
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          pattern = '<%sd'%length
          start = end
          end += struct.calcsize(pattern)
          val2.data = struct.unpack(pattern, str[start:end])
          val1.pose.append(val2)
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        val1.value = []
        for i in range(0, length):
          val2 = baxter_core_msgs.msg.FloatArray()
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          pattern = '<%sd'%length
          start = end
          end += struct.calcsize(pattern)
          val2.data = struct.unpack(pattern, str[start:end])
          val1.value.append(val2)
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        val1.rotation = []
        for i in range(0, length):
          val2 = baxter_core_msgs.msg.FloatArray()
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          pattern = '<%sd'%length
          start = end
          end += struct.calcsize(pattern)
          val2.data = struct.unpack(pattern, str[start:end])
          val1.rotation.append(val2)
        self.parameters.append(val1)
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.predicates = []
      for i in range(0, length):
        val1 = baxter_core_msgs.msg.Predicate()
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          val1.type = str[start:end].decode('utf-8')
        else:
          val1.type = str[start:end]
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          val1.name = str[start:end].decode('utf-8')
        else:
          val1.name = str[start:end]
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        val1.params = []
        for i in range(0, length):
          val2 = baxter_core_msgs.msg.Parameter()
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          start = end
          end += length
          if python3:
            val2.type = str[start:end].decode('utf-8')
          else:
            val2.type = str[start:end]
          start = end
          end += 1
          (val2.is_symbol,) = _struct_B.unpack(str[start:end])
          val2.is_symbol = bool(val2.is_symbol)
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          val2.lArmPose = []
          for i in range(0, length):
            val3 = baxter_core_msgs.msg.FloatArray()
            start = end
            end += 4
            (length,) = _struct_I.unpack(str[start:end])
            pattern = '<%sd'%length
            start = end
            end += struct.calcsize(pattern)
            val3.data = struct.unpack(pattern, str[start:end])
            val2.lArmPose.append(val3)
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          val2.rArmPose = []
          for i in range(0, length):
            val3 = baxter_core_msgs.msg.FloatArray()
            start = end
            end += 4
            (length,) = _struct_I.unpack(str[start:end])
            pattern = '<%sd'%length
            start = end
            end += struct.calcsize(pattern)
            val3.data = struct.unpack(pattern, str[start:end])
            val2.rArmPose.append(val3)
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          val2.lGripper = []
          for i in range(0, length):
            val3 = baxter_core_msgs.msg.FloatArray()
            start = end
            end += 4
            (length,) = _struct_I.unpack(str[start:end])
            pattern = '<%sd'%length
            start = end
            end += struct.calcsize(pattern)
            val3.data = struct.unpack(pattern, str[start:end])
            val2.lGripper.append(val3)
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          val2.rGripper = []
          for i in range(0, length):
            val3 = baxter_core_msgs.msg.FloatArray()
            start = end
            end += 4
            (length,) = _struct_I.unpack(str[start:end])
            pattern = '<%sd'%length
            start = end
            end += struct.calcsize(pattern)
            val3.data = struct.unpack(pattern, str[start:end])
            val2.rGripper.append(val3)
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          val2.pose = []
          for i in range(0, length):
            val3 = baxter_core_msgs.msg.FloatArray()
            start = end
            end += 4
            (length,) = _struct_I.unpack(str[start:end])
            pattern = '<%sd'%length
            start = end
            end += struct.calcsize(pattern)
            val3.data = struct.unpack(pattern, str[start:end])
            val2.pose.append(val3)
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          val2.value = []
          for i in range(0, length):
            val3 = baxter_core_msgs.msg.FloatArray()
            start = end
            end += 4
            (length,) = _struct_I.unpack(str[start:end])
            pattern = '<%sd'%length
            start = end
            end += struct.calcsize(pattern)
            val3.data = struct.unpack(pattern, str[start:end])
            val2.value.append(val3)
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          val2.rotation = []
          for i in range(0, length):
            val3 = baxter_core_msgs.msg.FloatArray()
            start = end
            end += 4
            (length,) = _struct_I.unpack(str[start:end])
            pattern = '<%sd'%length
            start = end
            end += struct.calcsize(pattern)
            val3.data = struct.unpack(pattern, str[start:end])
            val2.rotation.append(val3)
          val1.params.append(val2)
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        val1.param_types = []
        for i in range(0, length):
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          start = end
          end += length
          if python3:
            val2 = str[start:end].decode('utf-8')
          else:
            val2 = str[start:end]
          val1.param_types.append(val2)
        self.predicates.append(val1)
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e) #most likely buffer underfill


  def serialize_numpy(self, buff, numpy):
    """
    serialize message with numpy array types into buffer
    :param buff: buffer, ``StringIO``
    :param numpy: numpy python module
    """
    try:
      buff.write(_struct_i.pack(self.step_num))
      _x = self.name
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      if python3:
        buff.write(struct.pack('<I%sB'%length, length, *_x))
      else:
        buff.write(struct.pack('<I%ss'%length, length, _x))
      length = len(self.active_timesteps)
      buff.write(_struct_I.pack(length))
      pattern = '<%si'%length
      buff.write(self.active_timesteps.tostring())
      length = len(self.parameters)
      buff.write(_struct_I.pack(length))
      for val1 in self.parameters:
        _x = val1.type
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        if python3:
          buff.write(struct.pack('<I%sB'%length, length, *_x))
        else:
          buff.write(struct.pack('<I%ss'%length, length, _x))
        buff.write(_struct_B.pack(val1.is_symbol))
        length = len(val1.lArmPose)
        buff.write(_struct_I.pack(length))
        for val2 in val1.lArmPose:
          length = len(val2.data)
          buff.write(_struct_I.pack(length))
          pattern = '<%sd'%length
          buff.write(val2.data.tostring())
        length = len(val1.rArmPose)
        buff.write(_struct_I.pack(length))
        for val2 in val1.rArmPose:
          length = len(val2.data)
          buff.write(_struct_I.pack(length))
          pattern = '<%sd'%length
          buff.write(val2.data.tostring())
        length = len(val1.lGripper)
        buff.write(_struct_I.pack(length))
        for val2 in val1.lGripper:
          length = len(val2.data)
          buff.write(_struct_I.pack(length))
          pattern = '<%sd'%length
          buff.write(val2.data.tostring())
        length = len(val1.rGripper)
        buff.write(_struct_I.pack(length))
        for val2 in val1.rGripper:
          length = len(val2.data)
          buff.write(_struct_I.pack(length))
          pattern = '<%sd'%length
          buff.write(val2.data.tostring())
        length = len(val1.pose)
        buff.write(_struct_I.pack(length))
        for val2 in val1.pose:
          length = len(val2.data)
          buff.write(_struct_I.pack(length))
          pattern = '<%sd'%length
          buff.write(val2.data.tostring())
        length = len(val1.value)
        buff.write(_struct_I.pack(length))
        for val2 in val1.value:
          length = len(val2.data)
          buff.write(_struct_I.pack(length))
          pattern = '<%sd'%length
          buff.write(val2.data.tostring())
        length = len(val1.rotation)
        buff.write(_struct_I.pack(length))
        for val2 in val1.rotation:
          length = len(val2.data)
          buff.write(_struct_I.pack(length))
          pattern = '<%sd'%length
          buff.write(val2.data.tostring())
      length = len(self.predicates)
      buff.write(_struct_I.pack(length))
      for val1 in self.predicates:
        _x = val1.type
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        if python3:
          buff.write(struct.pack('<I%sB'%length, length, *_x))
        else:
          buff.write(struct.pack('<I%ss'%length, length, _x))
        _x = val1.name
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        if python3:
          buff.write(struct.pack('<I%sB'%length, length, *_x))
        else:
          buff.write(struct.pack('<I%ss'%length, length, _x))
        length = len(val1.params)
        buff.write(_struct_I.pack(length))
        for val2 in val1.params:
          _x = val2.type
          length = len(_x)
          if python3 or type(_x) == unicode:
            _x = _x.encode('utf-8')
            length = len(_x)
          if python3:
            buff.write(struct.pack('<I%sB'%length, length, *_x))
          else:
            buff.write(struct.pack('<I%ss'%length, length, _x))
          buff.write(_struct_B.pack(val2.is_symbol))
          length = len(val2.lArmPose)
          buff.write(_struct_I.pack(length))
          for val3 in val2.lArmPose:
            length = len(val3.data)
            buff.write(_struct_I.pack(length))
            pattern = '<%sd'%length
            buff.write(val3.data.tostring())
          length = len(val2.rArmPose)
          buff.write(_struct_I.pack(length))
          for val3 in val2.rArmPose:
            length = len(val3.data)
            buff.write(_struct_I.pack(length))
            pattern = '<%sd'%length
            buff.write(val3.data.tostring())
          length = len(val2.lGripper)
          buff.write(_struct_I.pack(length))
          for val3 in val2.lGripper:
            length = len(val3.data)
            buff.write(_struct_I.pack(length))
            pattern = '<%sd'%length
            buff.write(val3.data.tostring())
          length = len(val2.rGripper)
          buff.write(_struct_I.pack(length))
          for val3 in val2.rGripper:
            length = len(val3.data)
            buff.write(_struct_I.pack(length))
            pattern = '<%sd'%length
            buff.write(val3.data.tostring())
          length = len(val2.pose)
          buff.write(_struct_I.pack(length))
          for val3 in val2.pose:
            length = len(val3.data)
            buff.write(_struct_I.pack(length))
            pattern = '<%sd'%length
            buff.write(val3.data.tostring())
          length = len(val2.value)
          buff.write(_struct_I.pack(length))
          for val3 in val2.value:
            length = len(val3.data)
            buff.write(_struct_I.pack(length))
            pattern = '<%sd'%length
            buff.write(val3.data.tostring())
          length = len(val2.rotation)
          buff.write(_struct_I.pack(length))
          for val3 in val2.rotation:
            length = len(val3.data)
            buff.write(_struct_I.pack(length))
            pattern = '<%sd'%length
            buff.write(val3.data.tostring())
        length = len(val1.param_types)
        buff.write(_struct_I.pack(length))
        for val2 in val1.param_types:
          length = len(val2)
          if python3 or type(val2) == unicode:
            val2 = val2.encode('utf-8')
            length = len(val2)
          if python3:
            buff.write(struct.pack('<I%sB'%length, length, *val2))
          else:
            buff.write(struct.pack('<I%ss'%length, length, val2))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize_numpy(self, str, numpy):
    """
    unpack serialized message in str into this message instance using numpy for array types
    :param str: byte array of serialized message, ``str``
    :param numpy: numpy python module
    """
    try:
      if self.parameters is None:
        self.parameters = None
      if self.predicates is None:
        self.predicates = None
      end = 0
      start = end
      end += 4
      (self.step_num,) = _struct_i.unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.name = str[start:end].decode('utf-8')
      else:
        self.name = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%si'%length
      start = end
      end += struct.calcsize(pattern)
      self.active_timesteps = numpy.frombuffer(str[start:end], dtype=numpy.int32, count=length)
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.parameters = []
      for i in range(0, length):
        val1 = baxter_core_msgs.msg.Parameter()
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          val1.type = str[start:end].decode('utf-8')
        else:
          val1.type = str[start:end]
        start = end
        end += 1
        (val1.is_symbol,) = _struct_B.unpack(str[start:end])
        val1.is_symbol = bool(val1.is_symbol)
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        val1.lArmPose = []
        for i in range(0, length):
          val2 = baxter_core_msgs.msg.FloatArray()
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          pattern = '<%sd'%length
          start = end
          end += struct.calcsize(pattern)
          val2.data = numpy.frombuffer(str[start:end], dtype=numpy.float64, count=length)
          val1.lArmPose.append(val2)
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        val1.rArmPose = []
        for i in range(0, length):
          val2 = baxter_core_msgs.msg.FloatArray()
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          pattern = '<%sd'%length
          start = end
          end += struct.calcsize(pattern)
          val2.data = numpy.frombuffer(str[start:end], dtype=numpy.float64, count=length)
          val1.rArmPose.append(val2)
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        val1.lGripper = []
        for i in range(0, length):
          val2 = baxter_core_msgs.msg.FloatArray()
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          pattern = '<%sd'%length
          start = end
          end += struct.calcsize(pattern)
          val2.data = numpy.frombuffer(str[start:end], dtype=numpy.float64, count=length)
          val1.lGripper.append(val2)
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        val1.rGripper = []
        for i in range(0, length):
          val2 = baxter_core_msgs.msg.FloatArray()
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          pattern = '<%sd'%length
          start = end
          end += struct.calcsize(pattern)
          val2.data = numpy.frombuffer(str[start:end], dtype=numpy.float64, count=length)
          val1.rGripper.append(val2)
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        val1.pose = []
        for i in range(0, length):
          val2 = baxter_core_msgs.msg.FloatArray()
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          pattern = '<%sd'%length
          start = end
          end += struct.calcsize(pattern)
          val2.data = numpy.frombuffer(str[start:end], dtype=numpy.float64, count=length)
          val1.pose.append(val2)
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        val1.value = []
        for i in range(0, length):
          val2 = baxter_core_msgs.msg.FloatArray()
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          pattern = '<%sd'%length
          start = end
          end += struct.calcsize(pattern)
          val2.data = numpy.frombuffer(str[start:end], dtype=numpy.float64, count=length)
          val1.value.append(val2)
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        val1.rotation = []
        for i in range(0, length):
          val2 = baxter_core_msgs.msg.FloatArray()
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          pattern = '<%sd'%length
          start = end
          end += struct.calcsize(pattern)
          val2.data = numpy.frombuffer(str[start:end], dtype=numpy.float64, count=length)
          val1.rotation.append(val2)
        self.parameters.append(val1)
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.predicates = []
      for i in range(0, length):
        val1 = baxter_core_msgs.msg.Predicate()
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          val1.type = str[start:end].decode('utf-8')
        else:
          val1.type = str[start:end]
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          val1.name = str[start:end].decode('utf-8')
        else:
          val1.name = str[start:end]
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        val1.params = []
        for i in range(0, length):
          val2 = baxter_core_msgs.msg.Parameter()
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          start = end
          end += length
          if python3:
            val2.type = str[start:end].decode('utf-8')
          else:
            val2.type = str[start:end]
          start = end
          end += 1
          (val2.is_symbol,) = _struct_B.unpack(str[start:end])
          val2.is_symbol = bool(val2.is_symbol)
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          val2.lArmPose = []
          for i in range(0, length):
            val3 = baxter_core_msgs.msg.FloatArray()
            start = end
            end += 4
            (length,) = _struct_I.unpack(str[start:end])
            pattern = '<%sd'%length
            start = end
            end += struct.calcsize(pattern)
            val3.data = numpy.frombuffer(str[start:end], dtype=numpy.float64, count=length)
            val2.lArmPose.append(val3)
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          val2.rArmPose = []
          for i in range(0, length):
            val3 = baxter_core_msgs.msg.FloatArray()
            start = end
            end += 4
            (length,) = _struct_I.unpack(str[start:end])
            pattern = '<%sd'%length
            start = end
            end += struct.calcsize(pattern)
            val3.data = numpy.frombuffer(str[start:end], dtype=numpy.float64, count=length)
            val2.rArmPose.append(val3)
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          val2.lGripper = []
          for i in range(0, length):
            val3 = baxter_core_msgs.msg.FloatArray()
            start = end
            end += 4
            (length,) = _struct_I.unpack(str[start:end])
            pattern = '<%sd'%length
            start = end
            end += struct.calcsize(pattern)
            val3.data = numpy.frombuffer(str[start:end], dtype=numpy.float64, count=length)
            val2.lGripper.append(val3)
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          val2.rGripper = []
          for i in range(0, length):
            val3 = baxter_core_msgs.msg.FloatArray()
            start = end
            end += 4
            (length,) = _struct_I.unpack(str[start:end])
            pattern = '<%sd'%length
            start = end
            end += struct.calcsize(pattern)
            val3.data = numpy.frombuffer(str[start:end], dtype=numpy.float64, count=length)
            val2.rGripper.append(val3)
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          val2.pose = []
          for i in range(0, length):
            val3 = baxter_core_msgs.msg.FloatArray()
            start = end
            end += 4
            (length,) = _struct_I.unpack(str[start:end])
            pattern = '<%sd'%length
            start = end
            end += struct.calcsize(pattern)
            val3.data = numpy.frombuffer(str[start:end], dtype=numpy.float64, count=length)
            val2.pose.append(val3)
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          val2.value = []
          for i in range(0, length):
            val3 = baxter_core_msgs.msg.FloatArray()
            start = end
            end += 4
            (length,) = _struct_I.unpack(str[start:end])
            pattern = '<%sd'%length
            start = end
            end += struct.calcsize(pattern)
            val3.data = numpy.frombuffer(str[start:end], dtype=numpy.float64, count=length)
            val2.value.append(val3)
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          val2.rotation = []
          for i in range(0, length):
            val3 = baxter_core_msgs.msg.FloatArray()
            start = end
            end += 4
            (length,) = _struct_I.unpack(str[start:end])
            pattern = '<%sd'%length
            start = end
            end += struct.calcsize(pattern)
            val3.data = numpy.frombuffer(str[start:end], dtype=numpy.float64, count=length)
            val2.rotation.append(val3)
          val1.params.append(val2)
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        val1.param_types = []
        for i in range(0, length):
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          start = end
          end += length
          if python3:
            val2 = str[start:end].decode('utf-8')
          else:
            val2 = str[start:end]
          val1.param_types.append(val2)
        self.predicates.append(val1)
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e) #most likely buffer underfill

_struct_I = genpy.struct_I
_struct_i = struct.Struct("<i")
_struct_B = struct.Struct("<B")
