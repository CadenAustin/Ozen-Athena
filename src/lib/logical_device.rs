use anyhow::Result;
use std::collections::HashSet;

use vulkanalia::prelude::v1_0::*;

use crate::{
    app::{AppData, VALIDATION_ENABLED, VALIDATION_LAYER, PORTABILITY_MACOS_VERSION, DEVICE_EXTENSIONS},
    physical_device::QueueFamilyIndices,
};

pub(crate) unsafe fn create_logical_device(
  entry: &Entry,
  instance: &Instance,
  data: &mut AppData,
) -> Result<Device> {
  let indices = QueueFamilyIndices::get(instance, data, data.physical_device).unwrap();

  let mut unique_indices = HashSet::new();
  unique_indices.insert(indices.graphics);
  unique_indices.insert(indices.present);

  let queue_priorities = &[1.0];
  let queue_infos = unique_indices
      .iter()
      .map(|i| {
          vk::DeviceQueueCreateInfo::builder()
              .queue_family_index(*i)
              .queue_priorities(queue_priorities)
      })
      .collect::<Vec<_>>();

  let layers = if VALIDATION_ENABLED {
      vec![VALIDATION_LAYER.as_ptr()]
  } else {
      vec![]
  };

  let mut extensions = DEVICE_EXTENSIONS
      .iter()
      .map(|n| n.as_ptr())
      .collect::<Vec<_>>();

  if cfg!(target_os = "macos") && entry.version().unwrap() >= PORTABILITY_MACOS_VERSION {
      extensions.push(vk::KHR_PORTABILITY_SUBSET_EXTENSION.name.as_ptr());
  }

  let features = vk::PhysicalDeviceFeatures::builder().sampler_anisotropy(true);

  let info = vk::DeviceCreateInfo::builder()
      .queue_create_infos(&queue_infos)
      .enabled_layer_names(&layers)
      .enabled_extension_names(&extensions)
      .enabled_features(&features);

  let device = instance
      .create_device(data.physical_device, &info, None)
      .unwrap();

  data.graphics_queue = device.get_device_queue(indices.graphics, 0);
  data.present_queue = device.get_device_queue(indices.present, 0);

  Ok(device)
}