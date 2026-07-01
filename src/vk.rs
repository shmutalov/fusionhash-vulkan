//! Thin Vulkan compute wrapper built on `ash`. Headless (no surface/swapchain):
//! just what a compute miner needs — instance, physical-device enumeration,
//! a single compute queue, buffers, shader modules and pipelines.

use anyhow::{bail, Context, Result};
use ash::vk;
use std::ffi::{CStr, CString};
use std::sync::Arc;

pub const AMD_VENDOR_ID: u32 = 0x1002;
pub const NVIDIA_VENDOR_ID: u32 = 0x10DE;

pub struct Instance {
    // Kept alive so the Vulkan loader stays loaded for the process lifetime.
    #[allow(dead_code)]
    entry: ash::Entry,
    pub raw: ash::Instance,
}

impl Instance {
    pub fn new() -> Result<Arc<Self>> {
        let entry = unsafe { ash::Entry::load().context("failed to load the Vulkan loader")? };

        let app_name = CString::new("fusionhash-vulkan").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(&app_name)
            .api_version(vk::make_api_version(0, 1, 3, 0));

        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let raw = unsafe { entry.create_instance(&create_info, None) }
            .context("failed to create Vulkan instance")?;

        Ok(Arc::new(Self { entry, raw }))
    }

    /// Enumerate physical devices, sorted by PCI-ish stable order (handle order).
    pub fn enumerate(&self) -> Result<Vec<PhysicalDevice>> {
        let pds = unsafe { self.raw.enumerate_physical_devices() }
            .context("failed to enumerate physical devices")?;
        let mut out = Vec::new();
        for pd in pds {
            out.push(PhysicalDevice::query(&self.raw, pd));
        }
        Ok(out)
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe { self.raw.destroy_instance(None) };
    }
}

#[derive(Clone)]
pub struct PhysicalDevice {
    pub handle: vk::PhysicalDevice,
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_local_mem: u64,
    pub max_alloc: u64,
    pub compute_units: u32,
    pub subgroup_size: u32,
    pub driver_info: String,
    pub pci_bus: u32,
}

impl PhysicalDevice {
    fn query(instance: &ash::Instance, handle: vk::PhysicalDevice) -> Self {
        let props = unsafe { instance.get_physical_device_properties(handle) };
        let mem = unsafe { instance.get_physical_device_memory_properties(handle) };

        let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        let mut device_local_mem = 0u64;
        for i in 0..mem.memory_heap_count as usize {
            let heap = mem.memory_heaps[i];
            if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                device_local_mem = device_local_mem.max(heap.size);
            }
        }

        // Subgroup size + driver info + max allocation via the properties2 chain
        // (all Vulkan 1.1/1.2 core, always safe to query).
        let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
        let mut driver = vk::PhysicalDeviceDriverProperties::default();
        let mut maint3 = vk::PhysicalDeviceMaintenance3Properties::default();
        let mut pci = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut subgroup)
            .push_next(&mut driver)
            .push_next(&mut maint3)
            .push_next(&mut pci);
        unsafe { instance.get_physical_device_properties2(handle, &mut props2) };

        let driver_info = unsafe { CStr::from_ptr(driver.driver_info.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        let max_alloc = if maint3.max_memory_allocation_size == 0 {
            0x8000_0000
        } else {
            maint3.max_memory_allocation_size
        };

        // Compute-unit count: query the AMD shader-core properties when present.
        let mut compute_units = 0u32;
        if props.vendor_id == AMD_VENDOR_ID {
            let mut core = vk::PhysicalDeviceShaderCorePropertiesAMD::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut core);
            unsafe { instance.get_physical_device_properties2(handle, &mut p2) };
            compute_units = core.shader_engine_count
                * core.shader_arrays_per_engine_count
                * core.compute_units_per_shader_array;
        }

        Self {
            handle,
            name,
            vendor_id: props.vendor_id,
            device_id: props.device_id,
            device_local_mem,
            max_alloc,
            compute_units,
            subgroup_size: subgroup.subgroup_size,
            driver_info,
            pci_bus: pci.pci_bus,
        }
    }

    pub fn is_amd(&self) -> bool {
        self.vendor_id == AMD_VENDOR_ID
    }
    pub fn is_nvidia(&self) -> bool {
        self.vendor_id == NVIDIA_VENDOR_ID
    }
    pub fn is_gpu_vendor(&self) -> bool {
        self.is_amd() || self.is_nvidia()
    }
}

pub struct Gpu {
    // Kept alive so the logical device never outlives its instance.
    #[allow(dead_code)]
    pub instance: Arc<Instance>,
    pub pdev: PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub mem_props: vk::PhysicalDeviceMemoryProperties,
}

impl Gpu {
    pub fn new(instance: Arc<Instance>, pdev: PhysicalDevice) -> Result<Arc<Self>> {
        let raw = &instance.raw;

        // Pick a queue family that supports compute.
        let qf_props =
            unsafe { raw.get_physical_device_queue_family_properties(pdev.handle) };
        let queue_family = qf_props
            .iter()
            .enumerate()
            .find(|(_, p)| p.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(i, _)| i as u32)
            .context("no compute-capable queue family")?;

        let priorities = [1.0f32];
        let qci = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities);
        let qcis = [qci];

        let mut features = vk::PhysicalDeviceFeatures::default();
        features.shader_int64 = vk::TRUE;
        features.shader_float64 = vk::TRUE;

        // Vulkan 1.2 features: 8-bit/16-bit storage are not required, but we do
        // rely on shaderBufferInt64Atomics? No — plain uint atomics only.
        let dci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&qcis)
            .enabled_features(&features);

        let device = unsafe { raw.create_device(pdev.handle, &dci, None) }
            .context("failed to create logical device")?;
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let mem_props = unsafe { raw.get_physical_device_memory_properties(pdev.handle) };

        Ok(Arc::new(Self {
            instance,
            pdev,
            device,
            queue,
            queue_family,
            mem_props,
        }))
    }

    fn find_memory_type(&self, type_bits: u32, flags: vk::MemoryPropertyFlags) -> Result<u32> {
        for i in 0..self.mem_props.memory_type_count {
            let ok_type = (type_bits & (1 << i)) != 0;
            let ok_flags = self.mem_props.memory_types[i as usize]
                .property_flags
                .contains(flags);
            if ok_type && ok_flags {
                return Ok(i);
            }
        }
        bail!("no suitable memory type for flags {:?}", flags)
    }

    pub fn create_buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        host_visible: bool,
    ) -> Result<Buffer> {
        let ci = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { self.device.create_buffer(&ci, None) }
            .context("create_buffer failed")?;
        let req = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let flags = if host_visible {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        } else {
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        };
        let mem_type = self.find_memory_type(req.memory_type_bits, flags)?;

        let ai = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(mem_type);
        let memory = unsafe { self.device.allocate_memory(&ai, None) }
            .with_context(|| format!("allocate_memory failed for {size} bytes"))?;
        unsafe { self.device.bind_buffer_memory(buffer, memory, 0) }
            .context("bind_buffer_memory failed")?;

        let mapped = if host_visible {
            let ptr = unsafe {
                self.device
                    .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())?
            };
            ptr as *mut u8
        } else {
            std::ptr::null_mut()
        };

        Ok(Buffer {
            buffer,
            memory,
            size,
            mapped,
        })
    }

    pub fn destroy_buffer(&self, b: &Buffer) {
        unsafe {
            if !b.mapped.is_null() {
                self.device.unmap_memory(b.memory);
            }
            self.device.destroy_buffer(b.buffer, None);
            self.device.free_memory(b.memory, None);
        }
    }

    pub fn create_shader_module(&self, spv: &[u8]) -> Result<vk::ShaderModule> {
        let code = ash::util::read_spv(&mut std::io::Cursor::new(spv))
            .context("failed to read SPIR-V")?;
        let ci = vk::ShaderModuleCreateInfo::default().code(&code);
        Ok(unsafe { self.device.create_shader_module(&ci, None) }?)
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
        }
    }
}

pub struct Buffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    #[allow(dead_code)]
    pub size: u64,
    pub mapped: *mut u8,
}

impl Buffer {
    /// Copy `data` into a host-visible buffer.
    pub unsafe fn write_bytes(&self, offset: usize, data: &[u8]) {
        debug_assert!(!self.mapped.is_null());
        std::ptr::copy_nonoverlapping(data.as_ptr(), self.mapped.add(offset), data.len());
    }

    /// Read bytes from a host-visible buffer.
    pub unsafe fn read_bytes(&self, offset: usize, out: &mut [u8]) {
        debug_assert!(!self.mapped.is_null());
        std::ptr::copy_nonoverlapping(self.mapped.add(offset), out.as_mut_ptr(), out.len());
    }
}

/// Reinterpret a `#[repr(C)]` value as bytes for a push-constant upload.
pub fn as_bytes<T: Copy>(v: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v as *const T as *const u8, std::mem::size_of::<T>()) }
}
