//! Minimal hand-written Vulkan 1.3 compute bindings, loaded at run time
//! through `vkGetInstanceProcAddr` from `libvulkan.so.1` (Axiom 4: no
//! vendor SDK at build time). Only what headless compute needs is declared.
//! Struct layouts and enum values are transcribed from the Vulkan
//! specification (and cross-checked against `vulkan_core.h` during
//! development, which is never included by the build).

const std = @import("std");

pub const Result = i32;
pub const SUCCESS: Result = 0;
pub const TIMEOUT: Result = 2;
pub const ERROR_OUT_OF_HOST_MEMORY: Result = -1;
pub const ERROR_OUT_OF_DEVICE_MEMORY: Result = -2;
pub const ERROR_DEVICE_LOST: Result = -4;

pub const Instance = *opaque {};
pub const PhysicalDevice = *opaque {};
pub const Device = *opaque {};
pub const Queue = *opaque {};
pub const CommandBuffer = *opaque {};
pub const Handle = u64; // every non-dispatchable handle on 64-bit targets

pub const ST = struct {
    pub const application_info: u32 = 0;
    pub const instance_create_info: u32 = 1;
    pub const device_queue_create_info: u32 = 2;
    pub const device_create_info: u32 = 3;
    pub const submit_info: u32 = 4;
    pub const memory_allocate_info: u32 = 5;
    pub const fence_create_info: u32 = 8;
    pub const buffer_create_info: u32 = 12;
    pub const shader_module_create_info: u32 = 16;
    pub const pipeline_shader_stage_create_info: u32 = 18;
    pub const compute_pipeline_create_info: u32 = 29;
    pub const pipeline_layout_create_info: u32 = 30;
    pub const descriptor_set_layout_create_info: u32 = 32;
    pub const descriptor_pool_create_info: u32 = 33;
    pub const descriptor_set_allocate_info: u32 = 34;
    pub const write_descriptor_set: u32 = 35;
    pub const command_pool_create_info: u32 = 39;
    pub const command_buffer_allocate_info: u32 = 40;
    pub const command_buffer_begin_info: u32 = 42;
    pub const memory_barrier: u32 = 46;
    pub const physical_device_properties_2: u32 = 1000059001;
    pub const external_memory_buffer_create_info: u32 = 1000072000;
    pub const physical_device_8bit_storage_features: u32 = 1000177000;
    pub const physical_device_shader_float16_int8_features: u32 = 1000082000;
    pub const import_memory_host_pointer_info_ext: u32 = 1000178000;
    pub const memory_host_pointer_properties_ext: u32 = 1000178001;
    pub const physical_device_external_memory_host_properties_ext: u32 = 1000178002;
    pub const physical_device_float_controls_properties: u32 = 1000197000;
    pub const physical_device_cooperative_matrix_features_khr: u32 = 1000506000;
    pub const cooperative_matrix_properties_khr: u32 = 1000506001;
};

pub const QUEUE_COMPUTE: u32 = 0x2;
pub const MEM_DEVICE_LOCAL: u32 = 0x1;
pub const MEM_HOST_VISIBLE: u32 = 0x2;
pub const MEM_HOST_COHERENT: u32 = 0x4;
pub const BUF_TRANSFER_SRC: u32 = 0x1;
pub const BUF_TRANSFER_DST: u32 = 0x2;
pub const BUF_STORAGE: u32 = 0x20;
pub const SHARING_EXCLUSIVE: u32 = 0;
pub const DESCRIPTOR_STORAGE_BUFFER: u32 = 7;
pub const STAGE_COMPUTE: u32 = 0x20;
pub const PIPELINE_BIND_COMPUTE: u32 = 1;
pub const CMD_LEVEL_PRIMARY: u32 = 0;
pub const CMD_ONE_TIME_SUBMIT: u32 = 0x1;
pub const PIPE_STAGE_TRANSFER: u32 = 0x1000;
pub const PIPE_STAGE_COMPUTE: u32 = 0x800;
pub const PIPE_STAGE_HOST: u32 = 0x4000;
pub const ACCESS_SHADER_READ: u32 = 0x20;
pub const ACCESS_SHADER_WRITE: u32 = 0x40;
pub const ACCESS_TRANSFER_READ: u32 = 0x800;
pub const ACCESS_TRANSFER_WRITE: u32 = 0x1000;
pub const ACCESS_HOST_READ: u32 = 0x2000;
pub const WHOLE_SIZE: u64 = ~@as(u64, 0);
pub const HANDLE_HOST_ALLOCATION: u32 = 0x80;

pub const ApplicationInfo = extern struct {
    sType: u32 = ST.application_info,
    pNext: ?*const anyopaque = null,
    pApplicationName: ?[*:0]const u8 = null,
    applicationVersion: u32 = 0,
    pEngineName: ?[*:0]const u8 = null,
    engineVersion: u32 = 0,
    apiVersion: u32,
};

pub const InstanceCreateInfo = extern struct {
    sType: u32 = ST.instance_create_info,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    pApplicationInfo: ?*const ApplicationInfo,
    enabledLayerCount: u32 = 0,
    ppEnabledLayerNames: ?[*]const [*:0]const u8 = null,
    enabledExtensionCount: u32 = 0,
    ppEnabledExtensionNames: ?[*]const [*:0]const u8 = null,
};

pub const QueueFamilyProperties = extern struct {
    queueFlags: u32,
    queueCount: u32,
    timestampValidBits: u32,
    granularity: [3]u32,
};

pub const DeviceQueueCreateInfo = extern struct {
    sType: u32 = ST.device_queue_create_info,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    queueFamilyIndex: u32,
    queueCount: u32 = 1,
    pQueuePriorities: [*]const f32,
};

pub const DeviceCreateInfo = extern struct {
    sType: u32 = ST.device_create_info,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    queueCreateInfoCount: u32 = 1,
    pQueueCreateInfos: [*]const DeviceQueueCreateInfo,
    enabledLayerCount: u32 = 0,
    ppEnabledLayerNames: ?[*]const [*:0]const u8 = null,
    enabledExtensionCount: u32 = 0,
    ppEnabledExtensionNames: ?[*]const [*:0]const u8 = null,
    pEnabledFeatures: ?*const anyopaque = null,
};

pub const ExtensionProperties = extern struct { name: [256]u8, specVersion: u32 };

pub const MemoryType = extern struct { propertyFlags: u32, heapIndex: u32 };
pub const MemoryHeap = extern struct { size: u64, flags: u32 };
pub const PhysicalDeviceMemoryProperties = extern struct {
    memoryTypeCount: u32,
    memoryTypes: [32]MemoryType,
    memoryHeapCount: u32,
    memoryHeaps: [16]MemoryHeap,
};

pub const PhysicalDeviceProperties2 = extern struct {
    sType: u32 = ST.physical_device_properties_2,
    pNext: ?*anyopaque = null,
    // VkPhysicalDeviceProperties (824 bytes on LP64): apiVersion @0,
    // vendorID @8, deviceID @12, deviceName @20; the rest is slack.
    properties: [1024]u8 align(8) = [_]u8{0} ** 1024,
};

pub const FloatControlsProperties = extern struct {
    sType: u32 = ST.physical_device_float_controls_properties,
    pNext: ?*anyopaque = null,
    denormBehaviorIndependence: u32 = 0,
    roundingModeIndependence: u32 = 0,
    flags: [15]u32 = [_]u32{0} ** 15, // [4] = shaderDenormPreserveFloat32
};

pub const ExternalMemoryHostProperties = extern struct {
    sType: u32 = ST.physical_device_external_memory_host_properties_ext,
    pNext: ?*anyopaque = null,
    minImportedHostPointerAlignment: u64 = 0,
};

pub const CooperativeMatrixProperties = extern struct {
    sType: u32 = ST.cooperative_matrix_properties_khr,
    pNext: ?*anyopaque = null,
    MSize: u32 = 0,
    NSize: u32 = 0,
    KSize: u32 = 0,
    AType: u32 = 0,
    BType: u32 = 0,
    CType: u32 = 0,
    ResultType: u32 = 0,
    saturatingAccumulation: u32 = 0,
    scope: u32 = 0,
};

pub const CooperativeMatrixFeatures = extern struct {
    sType: u32 = ST.physical_device_cooperative_matrix_features_khr,
    pNext: ?*const anyopaque = null,
    cooperativeMatrix: u32 = 1,
    cooperativeMatrixRobustBufferAccess: u32 = 0,
};

pub const Storage8BitFeatures = extern struct {
    sType: u32 = ST.physical_device_8bit_storage_features,
    pNext: ?*const anyopaque = null,
    storageBuffer8BitAccess: u32 = 1,
    uniformAndStorageBuffer8BitAccess: u32 = 0,
    storagePushConstant8: u32 = 0,
};

pub const Float16Int8Features = extern struct {
    sType: u32 = ST.physical_device_shader_float16_int8_features,
    pNext: ?*const anyopaque = null,
    shaderFloat16: u32 = 0,
    shaderInt8: u32 = 1,
};

pub const BufferCreateInfo = extern struct {
    sType: u32 = ST.buffer_create_info,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    size: u64,
    usage: u32,
    sharingMode: u32 = SHARING_EXCLUSIVE,
    queueFamilyIndexCount: u32 = 0,
    pQueueFamilyIndices: ?[*]const u32 = null,
};

pub const ExternalMemoryBufferCreateInfo = extern struct {
    sType: u32 = ST.external_memory_buffer_create_info,
    pNext: ?*const anyopaque = null,
    handleTypes: u32 = HANDLE_HOST_ALLOCATION,
};

pub const MemoryRequirements = extern struct { size: u64, alignment: u64, memoryTypeBits: u32 };

pub const MemoryAllocateInfo = extern struct {
    sType: u32 = ST.memory_allocate_info,
    pNext: ?*const anyopaque = null,
    allocationSize: u64,
    memoryTypeIndex: u32,
};

pub const ImportMemoryHostPointerInfo = extern struct {
    sType: u32 = ST.import_memory_host_pointer_info_ext,
    pNext: ?*const anyopaque = null,
    handleType: u32 = HANDLE_HOST_ALLOCATION,
    pHostPointer: *anyopaque,
};

pub const MemoryHostPointerProperties = extern struct {
    sType: u32 = ST.memory_host_pointer_properties_ext,
    pNext: ?*anyopaque = null,
    memoryTypeBits: u32 = 0,
};

pub const ShaderModuleCreateInfo = extern struct {
    sType: u32 = ST.shader_module_create_info,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    codeSize: usize,
    pCode: [*]const u32,
};

pub const DescriptorSetLayoutBinding = extern struct {
    binding: u32,
    descriptorType: u32 = DESCRIPTOR_STORAGE_BUFFER,
    descriptorCount: u32 = 1,
    stageFlags: u32 = STAGE_COMPUTE,
    pImmutableSamplers: ?*const anyopaque = null,
};

pub const DescriptorSetLayoutCreateInfo = extern struct {
    sType: u32 = ST.descriptor_set_layout_create_info,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    bindingCount: u32,
    pBindings: [*]const DescriptorSetLayoutBinding,
};

pub const PushConstantRange = extern struct { stageFlags: u32 = STAGE_COMPUTE, offset: u32 = 0, size: u32 };

pub const PipelineLayoutCreateInfo = extern struct {
    sType: u32 = ST.pipeline_layout_create_info,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    setLayoutCount: u32 = 1,
    pSetLayouts: [*]const Handle,
    pushConstantRangeCount: u32,
    pPushConstantRanges: [*]const PushConstantRange,
};

pub const PipelineShaderStageCreateInfo = extern struct {
    sType: u32 = ST.pipeline_shader_stage_create_info,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    stage: u32 = STAGE_COMPUTE,
    module: Handle,
    pName: [*:0]const u8 = "main",
    pSpecializationInfo: ?*const anyopaque = null,
};

pub const ComputePipelineCreateInfo = extern struct {
    sType: u32 = ST.compute_pipeline_create_info,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    stage: PipelineShaderStageCreateInfo,
    layout: Handle,
    basePipelineHandle: Handle = 0,
    basePipelineIndex: i32 = -1,
};

pub const DescriptorPoolSize = extern struct { type: u32 = DESCRIPTOR_STORAGE_BUFFER, descriptorCount: u32 };

pub const DescriptorPoolCreateInfo = extern struct {
    sType: u32 = ST.descriptor_pool_create_info,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    maxSets: u32,
    poolSizeCount: u32 = 1,
    pPoolSizes: [*]const DescriptorPoolSize,
};

pub const DescriptorSetAllocateInfo = extern struct {
    sType: u32 = ST.descriptor_set_allocate_info,
    pNext: ?*const anyopaque = null,
    descriptorPool: Handle,
    descriptorSetCount: u32 = 1,
    pSetLayouts: [*]const Handle,
};

pub const DescriptorBufferInfo = extern struct { buffer: Handle, offset: u64 = 0, range: u64 = WHOLE_SIZE };

pub const WriteDescriptorSet = extern struct {
    sType: u32 = ST.write_descriptor_set,
    pNext: ?*const anyopaque = null,
    dstSet: Handle,
    dstBinding: u32,
    dstArrayElement: u32 = 0,
    descriptorCount: u32 = 1,
    descriptorType: u32 = DESCRIPTOR_STORAGE_BUFFER,
    pImageInfo: ?*const anyopaque = null,
    pBufferInfo: *const DescriptorBufferInfo,
    pTexelBufferView: ?*const anyopaque = null,
};

pub const CommandPoolCreateInfo = extern struct {
    sType: u32 = ST.command_pool_create_info,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    queueFamilyIndex: u32,
};

pub const CommandBufferAllocateInfo = extern struct {
    sType: u32 = ST.command_buffer_allocate_info,
    pNext: ?*const anyopaque = null,
    commandPool: Handle,
    level: u32 = CMD_LEVEL_PRIMARY,
    commandBufferCount: u32 = 1,
};

pub const CommandBufferBeginInfo = extern struct {
    sType: u32 = ST.command_buffer_begin_info,
    pNext: ?*const anyopaque = null,
    flags: u32 = CMD_ONE_TIME_SUBMIT,
    pInheritanceInfo: ?*const anyopaque = null,
};

pub const MemoryBarrier = extern struct {
    sType: u32 = ST.memory_barrier,
    pNext: ?*const anyopaque = null,
    srcAccessMask: u32,
    dstAccessMask: u32,
};

pub const BufferCopy = extern struct { srcOffset: u64, dstOffset: u64, size: u64 };

pub const SubmitInfo = extern struct {
    sType: u32 = ST.submit_info,
    pNext: ?*const anyopaque = null,
    waitSemaphoreCount: u32 = 0,
    pWaitSemaphores: ?*const anyopaque = null,
    pWaitDstStageMask: ?*const anyopaque = null,
    commandBufferCount: u32 = 1,
    pCommandBuffers: [*]const CommandBuffer,
    signalSemaphoreCount: u32 = 0,
    pSignalSemaphores: ?*const anyopaque = null,
};

pub const FenceCreateInfo = extern struct { sType: u32 = ST.fence_create_info, pNext: ?*const anyopaque = null, flags: u32 = 0 };

// ------------------------------------------------------------ dispatch --

const cc = std.builtin.CallingConvention.c;
pub const PFN = *const fn () callconv(cc) void;

/// Every entry point vapor uses, resolved once.
pub const Fns = struct {
    getInstanceProcAddr: *const fn (?Instance, [*:0]const u8) callconv(cc) ?PFN,
    createInstance: *const fn (*const InstanceCreateInfo, ?*const anyopaque, *?Instance) callconv(cc) Result = undefined,
    destroyInstance: *const fn (Instance, ?*const anyopaque) callconv(cc) void = undefined,
    enumeratePhysicalDevices: *const fn (Instance, *u32, ?[*]PhysicalDevice) callconv(cc) Result = undefined,
    getPhysicalDeviceProperties2: *const fn (PhysicalDevice, *PhysicalDeviceProperties2) callconv(cc) void = undefined,
    getPhysicalDeviceQueueFamilyProperties: *const fn (PhysicalDevice, *u32, ?[*]QueueFamilyProperties) callconv(cc) void = undefined,
    getPhysicalDeviceMemoryProperties: *const fn (PhysicalDevice, *PhysicalDeviceMemoryProperties) callconv(cc) void = undefined,
    enumerateDeviceExtensionProperties: *const fn (PhysicalDevice, ?[*:0]const u8, *u32, ?[*]ExtensionProperties) callconv(cc) Result = undefined,
    getCoopMatrixProperties: ?*const fn (PhysicalDevice, *u32, ?[*]CooperativeMatrixProperties) callconv(cc) Result = null,
    createDevice: *const fn (PhysicalDevice, *const DeviceCreateInfo, ?*const anyopaque, *?Device) callconv(cc) Result = undefined,
    getDeviceProcAddr: *const fn (Device, [*:0]const u8) callconv(cc) ?PFN = undefined,

    // device level
    destroyDevice: *const fn (Device, ?*const anyopaque) callconv(cc) void = undefined,
    getDeviceQueue: *const fn (Device, u32, u32, *?Queue) callconv(cc) void = undefined,
    createBuffer: *const fn (Device, *const BufferCreateInfo, ?*const anyopaque, *Handle) callconv(cc) Result = undefined,
    destroyBuffer: *const fn (Device, Handle, ?*const anyopaque) callconv(cc) void = undefined,
    getBufferMemoryRequirements: *const fn (Device, Handle, *MemoryRequirements) callconv(cc) void = undefined,
    allocateMemory: *const fn (Device, *const MemoryAllocateInfo, ?*const anyopaque, *Handle) callconv(cc) Result = undefined,
    freeMemory: *const fn (Device, Handle, ?*const anyopaque) callconv(cc) void = undefined,
    bindBufferMemory: *const fn (Device, Handle, Handle, u64) callconv(cc) Result = undefined,
    mapMemory: *const fn (Device, Handle, u64, u64, u32, *?*anyopaque) callconv(cc) Result = undefined,
    getMemoryHostPointerProperties: ?*const fn (Device, u32, *const anyopaque, *MemoryHostPointerProperties) callconv(cc) Result = null,
    createShaderModule: *const fn (Device, *const ShaderModuleCreateInfo, ?*const anyopaque, *Handle) callconv(cc) Result = undefined,
    destroyShaderModule: *const fn (Device, Handle, ?*const anyopaque) callconv(cc) void = undefined,
    createDescriptorSetLayout: *const fn (Device, *const DescriptorSetLayoutCreateInfo, ?*const anyopaque, *Handle) callconv(cc) Result = undefined,
    destroyDescriptorSetLayout: *const fn (Device, Handle, ?*const anyopaque) callconv(cc) void = undefined,
    createPipelineLayout: *const fn (Device, *const PipelineLayoutCreateInfo, ?*const anyopaque, *Handle) callconv(cc) Result = undefined,
    destroyPipelineLayout: *const fn (Device, Handle, ?*const anyopaque) callconv(cc) void = undefined,
    createComputePipelines: *const fn (Device, Handle, u32, [*]const ComputePipelineCreateInfo, ?*const anyopaque, [*]Handle) callconv(cc) Result = undefined,
    destroyPipeline: *const fn (Device, Handle, ?*const anyopaque) callconv(cc) void = undefined,
    createDescriptorPool: *const fn (Device, *const DescriptorPoolCreateInfo, ?*const anyopaque, *Handle) callconv(cc) Result = undefined,
    destroyDescriptorPool: *const fn (Device, Handle, ?*const anyopaque) callconv(cc) void = undefined,
    allocateDescriptorSets: *const fn (Device, *const DescriptorSetAllocateInfo, *Handle) callconv(cc) Result = undefined,
    updateDescriptorSets: *const fn (Device, u32, [*]const WriteDescriptorSet, u32, ?*const anyopaque) callconv(cc) void = undefined,
    createCommandPool: *const fn (Device, *const CommandPoolCreateInfo, ?*const anyopaque, *Handle) callconv(cc) Result = undefined,
    destroyCommandPool: *const fn (Device, Handle, ?*const anyopaque) callconv(cc) void = undefined,
    allocateCommandBuffers: *const fn (Device, *const CommandBufferAllocateInfo, *?CommandBuffer) callconv(cc) Result = undefined,
    beginCommandBuffer: *const fn (CommandBuffer, *const CommandBufferBeginInfo) callconv(cc) Result = undefined,
    endCommandBuffer: *const fn (CommandBuffer) callconv(cc) Result = undefined,
    cmdBindPipeline: *const fn (CommandBuffer, u32, Handle) callconv(cc) void = undefined,
    cmdBindDescriptorSets: *const fn (CommandBuffer, u32, Handle, u32, u32, [*]const Handle, u32, ?[*]const u32) callconv(cc) void = undefined,
    cmdPushConstants: *const fn (CommandBuffer, Handle, u32, u32, u32, *const anyopaque) callconv(cc) void = undefined,
    cmdDispatch: *const fn (CommandBuffer, u32, u32, u32) callconv(cc) void = undefined,
    cmdPipelineBarrier: *const fn (CommandBuffer, u32, u32, u32, u32, ?[*]const MemoryBarrier, u32, ?*const anyopaque, u32, ?*const anyopaque) callconv(cc) void = undefined,
    cmdCopyBuffer: *const fn (CommandBuffer, Handle, Handle, u32, [*]const BufferCopy) callconv(cc) void = undefined,
    cmdFillBuffer: *const fn (CommandBuffer, Handle, u64, u64, u32) callconv(cc) void = undefined,
    freeCommandBuffers: *const fn (Device, Handle, u32, [*]const CommandBuffer) callconv(cc) void = undefined,
    queueSubmit: *const fn (Queue, u32, [*]const SubmitInfo, Handle) callconv(cc) Result = undefined,
    createFence: *const fn (Device, *const FenceCreateInfo, ?*const anyopaque, *Handle) callconv(cc) Result = undefined,
    destroyFence: *const fn (Device, Handle, ?*const anyopaque) callconv(cc) void = undefined,
    waitForFences: *const fn (Device, u32, [*]const Handle, u32, u64) callconv(cc) Result = undefined,
    resetFences: *const fn (Device, u32, [*]const Handle) callconv(cc) Result = undefined,

    pub fn loadInstance(self: *Fns, inst: Instance) !void {
        inline for (.{
            .{ "destroyInstance", "vkDestroyInstance" },
            .{ "enumeratePhysicalDevices", "vkEnumeratePhysicalDevices" },
            .{ "getPhysicalDeviceProperties2", "vkGetPhysicalDeviceProperties2" },
            .{ "getPhysicalDeviceQueueFamilyProperties", "vkGetPhysicalDeviceQueueFamilyProperties" },
            .{ "getPhysicalDeviceMemoryProperties", "vkGetPhysicalDeviceMemoryProperties" },
            .{ "enumerateDeviceExtensionProperties", "vkEnumerateDeviceExtensionProperties" },
            .{ "createDevice", "vkCreateDevice" },
            .{ "getDeviceProcAddr", "vkGetDeviceProcAddr" },
        }) |p| {
            const f = self.getInstanceProcAddr(inst, p[1]) orelse return error.MissingEntryPoint;
            @field(self, p[0]) = @ptrCast(f);
        }
        if (self.getInstanceProcAddr(inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR")) |f|
            self.getCoopMatrixProperties = @ptrCast(f);
    }

    pub fn loadDevice(self: *Fns, dev: Device) !void {
        @setEvalBranchQuota(20000);
        inline for (@typeInfo(Fns).@"struct".fields) |fld| {
            if (comptime isDeviceLevel(fld.name)) {
                const name = comptime "vk" ++ [_]u8{std.ascii.toUpper(fld.name[0])} ++ fld.name[1..];
                const f = self.getDeviceProcAddr(dev, name) orelse return error.MissingEntryPoint;
                @field(self, fld.name) = @ptrCast(f);
            }
        }
        if (self.getDeviceProcAddr(dev, "vkGetMemoryHostPointerPropertiesEXT")) |f|
            self.getMemoryHostPointerProperties = @ptrCast(f);
    }

    fn isDeviceLevel(comptime name: []const u8) bool {
        const instance_level = [_][]const u8{
            "getInstanceProcAddr",                "createInstance",                     "destroyInstance",
            "enumeratePhysicalDevices",           "getPhysicalDeviceProperties2",       "getPhysicalDeviceQueueFamilyProperties",
            "getPhysicalDeviceMemoryProperties",  "enumerateDeviceExtensionProperties", "getCoopMatrixProperties",
            "createDevice",                       "getDeviceProcAddr",                  "getMemoryHostPointerProperties",
        };
        for (instance_level) |n| if (std.mem.eql(u8, n, name)) return false;
        return true;
    }
};
