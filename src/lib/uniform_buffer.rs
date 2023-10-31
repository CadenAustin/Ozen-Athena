use anyhow::Result;
use std::mem::size_of;

use vulkanalia::prelude::v1_0::*;

use crate::{app::AppData, types::Mat4, vertex_buffer::create_buffer};

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub(crate) struct UniformBufferObject {
    pub(crate) view: Mat4,
    pub(crate) proj: Mat4,
}

pub(crate) unsafe fn create_uniform_buffers(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    data.uniform_buffers.clear();
    data.uniform_buffers_memory.clear();

    for _ in 0..data.swapchain_images.len() {
        let (uniform_buffer, uniform_buffer_memory) = create_buffer(
            instance,
            device,
            data,
            size_of::<UniformBufferObject>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )
        .unwrap();

        data.uniform_buffers.push(uniform_buffer);
        data.uniform_buffers_memory.push(uniform_buffer_memory);
    }

    Ok(())
}
