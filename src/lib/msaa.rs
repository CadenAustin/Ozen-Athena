use vulkanalia::prelude::v1_0::*;

use crate::app::AppData;

pub(crate) unsafe fn get_max_msaa_samples(instance: &Instance, data: &AppData) -> vk::SampleCountFlags {
  let properties = instance.get_physical_device_properties(data.physical_device);
  let counts = properties.limits.framebuffer_color_sample_counts
      & properties.limits.framebuffer_depth_sample_counts;
  [
      vk::SampleCountFlags::_64,
      vk::SampleCountFlags::_32,
      vk::SampleCountFlags::_16,
      vk::SampleCountFlags::_8,
      vk::SampleCountFlags::_4,
      vk::SampleCountFlags::_2,
  ]
  .iter()
  .cloned()
  .find(|c| counts.contains(*c))
  .unwrap_or(vk::SampleCountFlags::_1)
}