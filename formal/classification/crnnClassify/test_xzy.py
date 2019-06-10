
import torch
import dataset





if __name__ == "__main__":
  test_lmdb_path = "../../../../dataset_formal/classify_data/crnnData/trainDataLMDB"

  test_dataset = dataset.lmdbDataset(root=test_lmdb_path, type="test")
  train_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=64,
    shuffle=False,  # xzy
    num_workers=int(2),
    collate_fn=dataset.alignCollate(imgH=32, imgW=100, keep_ratio=opt.keep_ratio))


  #
  # train_iter = iter(train_loader)
  # i = 0
  # while i < len(train_loader):
  #     for p in crnn.parameters():
  #         p.requires_grad = True
  #     crnn.train()
  #
  #     cost = trainBatch(crnn, criterion, optimizer)
  #     loss_avg.add(cost)
  #     i += 1
  #
  #     if i % opt.displayInterval == 0:
  #         print('[%d/%d][%d/%d] Loss: %f' %
  #               (epoch, opt.nepoch, i, len(train_loader), loss_avg.val()))
  #         loss_avg.reset()
  #
  #     if i % opt.valInterval == 0:
  #         val(crnn, test_dataset, criterion)
  #
  #     # do checkpointing
  #     if i % opt.saveInterval == 0:
  #         torch.save(
  #             crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(opt.expr_dir, epoch, i))
