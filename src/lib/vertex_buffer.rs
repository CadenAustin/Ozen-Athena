use anyhow::{anyhow, Result};
use std::{
  ptr::copy_nonoverlapping as memcpy,
  mem::size_of
};

use vulkanalia::prelude::v1_0::*;

use crate::{
  app::AppData,
  vertex::Vertex,
  single_time_cmd::{begin_single_time_commands, end_single_time_commands}
};

pub(crate) unsafe fn create_buffer(
  instance: &Instance,
  device: &Device,
  data: &AppData,
  size: vk::DeviceSize,
  usage: vk::BufferUsageFlags,
  properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
  let buffer_info = vk::BufferCreateInfo::builder()
      .size(size)
      .usage(usage)
      .sharing_mode(vk::SharingMode::EXCLUSIVE);

  let buffer = device.create_buffer(&buffer_info, None).unwrap();
  let requirements = device.get_buffer_memory_requirements(buffer);

  let memory_info = vk::MemoryAllocateInfo::builder()
      .allocation_size(requirements.size)
      .memory_type_index(
          get_memory_type_index(instance, data, properties, requirements).unwrap(),
      );
  let buffer_memory = device.allocate_memory(&memory_info, None).unwrap();

  device.bind_buffer_memory(buffer, buffer_memory, 0).unwrap();

  Ok((buffer, buffer_memory))
}

pub(crate) unsafe fn create_vertex_buffer(
  instance: &Instance,
  device: &Device,
  data: &mut AppData,
) -> Result<()> {
  let size = (size_of::<Vertex>() * data.vertices.len()) as u64;

  let (staging_buffer, staging_buffer_memory) = create_buffer(
      instance,
      device,
      data,
      size,
      vk::BufferUsageFlags::TRANSFER_SRC,
      vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
  )
  .unwrap();

  let memory = device
      .map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())
      .unwrap();

  memcpy(data.vertices.as_ptr(), memory.cast(), data.vertices.len());

  device.unmap_memory(staging_buffer_memory);

  let (vertex_buffer, vertex_buffer_memory) = create_buffer(
      instance,
      device,
      data,
      size,
      vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
      vk::MemoryPropertyFlags::DEVICE_LOCAL,
  )
  .unwrap();

  data.vertex_buffer = vertex_buffer;
  data.vertex_buffer_memory = vertex_buffer_memory;

  copy_buffer(device, data, staging_buffer, vertex_buffer, size).unwrap();

  device.destroy_buffer(staging_buffer, None);
  device.free_memory(staging_buffer_memory, None);

  Ok(())
}

pub(crate) unsafe fn create_index_buffer(
  instance: &Instance,
  device: &Device,
  data: &mut AppData,
) -> Result<()> {
  let size = (size_of::<u32>() * data.indices.len()) as u64;

  let (staging_buffer, staging_buffer_memory) = create_buffer(
      instance,
      device,
      data,
      size,
      vk::BufferUsageFlags::TRANSFER_SRC,
      vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
  )
  .unwrap();

  let memory = device
      .map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())
      .unwrap();

  memcpy(data.indices.as_ptr(), memory.cast(), data.indices.len());

  device.unmap_memory(staging_buffer_memory);

  let (index_buffer, index_buffer_memory) = create_buffer(
      instance,
      device,
      data,
      size,
      vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
      vk::MemoryPropertyFlags::DEVICE_LOCAL,
  )
  .unwrap();

  data.index_buffer = index_buffer;
  data.index_buffer_memory = index_buffer_memory;

  copy_buffer(device, data, staging_buffer, index_buffer, size).unwrap();

  device.destroy_buffer(staging_buffer, None);
  device.free_memory(staging_buffer_memory, None);

  Ok(())
}

pub(crate) unsafe fn get_memory_type_index(
  instance: &Instance,
  data: &AppData,
  properties: vk::MemoryPropertyFlags,
  requirements: vk::MemoryRequirements,
) -> Result<u32> {
  let memory = instance.get_physical_device_memory_properties(data.physical_device);

  (0..memory.memory_type_count)
      .find(|i| {
          let suitable = (requirements.memory_type_bits & (1 << i)) != 0;
          let memory_type = memory.memory_types[*i as usize];
          suitable && memory_type.property_flags.contains(properties)
      })
      .ok_or_else(|| anyhow!("Failed to find suitable memory type."))
}

pub(crate) unsafe fn copy_buffer(
  device: &Device,
  data: &AppData,
  source: vk::Buffer,
  destination: vk::Buffer,
  size: vk::DeviceSize,
) -> Result<()> {
  let command_buffer = begin_single_time_commands(device, data).unwrap();

  let regions = vk::BufferCopy::builder().size(size);
  device.cmd_copy_buffer(command_buffer, source, destination, &[regions]);

  end_single_time_commands(device, data, command_buffer).unwrap();
  Ok(())
}